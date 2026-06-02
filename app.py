import streamlit as st
import base64
import io
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lifestyle Image Generator",
    page_icon="📸",
    layout="wide",
)

# ── Gemini clients ────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    """Default client (v1beta) — used for text analysis and Gemini Flash image gen."""
    try:
        from google import genai
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except KeyError:
        return None

@st.cache_resource
def get_imagen_client():
    """v1 client — required for Imagen 3."""
    try:
        from google import genai
        from google.genai import types
        return genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"],
            http_options=types.HttpOptions(api_version="v1"),
        )
    except KeyError:
        return None

@st.cache_resource
def get_flash_image_client():
    """v1alpha client — required for experimental Gemini image generation models."""
    try:
        from google import genai
        from google.genai import types
        return genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"],
            http_options=types.HttpOptions(api_version="v1alpha"),
        )
    except KeyError:
        return None


# ── Constanten ────────────────────────────────────────────────────────────────
PRODUCT_TYPES = [
    "Powerbank",
    "Draadloze Oordopjes",
    "Smartphonehoesje",
    "Laptophoes",
    "Smartwatch",
    "Draadloze Speaker",
    "Anders",
]

# Sleutels = Nederlandse weergavenaam | Waarden = Engelse AI-prompt
SCENARIOS = {
    "Treinreis": (
        "placed flat on a textured plastic train tray-table next to a dark blue passport "
        "and a ceramic coffee mug. Natural window light casting soft shadows, blurred "
        "landscape rushing by in the background."
    ),
    "Muziekfestival": (
        "placed on a weathered wooden picnic table at a bustling outdoor music festival, "
        "next to a colorful festival wristband and trendy sunglasses. Casual lighting, "
        "blurred background of festival crowds and tents."
    ),
    "Stadspark": (
        "placed on a classic checkered picnic blanket in the lush green grass of a city "
        "park, next to a pair of sunglasses and a glass soda bottle. Warm, golden sunlight "
        "filtering through tree leaves."
    ),
    "Modern Kantoor": (
        "placed on a real oak wood desk next to a modern wireless computer mouse and a "
        "black ink pen. Clean, everyday office environment with realistic proportions and "
        "reflections."
    ),
    "Natuur / Buiten": (
        "placed on a rough, dusty grey rock next to a folded paper map and a standard "
        "metallic climbing carabiner. Diffused, overcast daylight in a pine forest."
    ),
    "✏️ Eigen scenario": None,
    "📷 Voorbeeldafbeelding": None,
}

DEFAULT_LIGHTING = (
    "Warm, bright, and natural tones. Soft golden hour sunlight or clean interior "
    "lighting. Avoid dark neon profiles."
)
DEFAULT_MOOD = (
    "Minimalist, clean, and modern lifestyle photography. Candid and authentic "
    "atmosphere. Premium but lived-in."
)
DEFAULT_NEGATIVE = (
    "trash, chaotic backgrounds, cheap-looking plastic, text, low-quality, CGI look, "
    "plastic texture, watermark, logo, brand name, blurry product."
)

SCENE_ANALYSIS_PROMPT = """You are a professional product photographer's assistant analyzing a reference scene image.
Describe the environment and setting visible in this image so it can be used as a backdrop for a product photo.

Focus ONLY on:
- The surface or foreground where a product could be placed (material, texture, color)
- Nearby props and objects visible in the scene
- Background environment and depth
- Lighting conditions (direction, quality, warm/cool, soft/hard)
- Overall atmosphere and mood

Do NOT describe any people, faces, or main subjects in the image.
Output a single concise paragraph starting with "placed on/in/near..." that describes
where and how a product would appear in this scene. No headers, no lists."""

ANALYSIS_PROMPT = """You are a professional product photographer's assistant.
Analyze the uploaded product image(s) and produce a hyper-detailed, technical,
unbranded visual description of the physical product only.

Include ALL of the following if visible:
- Overall shape and form factor (dimensions, thickness, curvature)
- Surface textures (matte, glossy, brushed metal, rubberized, etc.)
- Color(s) — use precise color names (e.g. "slate grey", "cream white")
- Visible ports, buttons, indicators, seams, and their placement
- Material quality cues (premium metal, soft-touch plastic, fabric, etc.)
- Any distinguishing physical features (rounded corners, ridges, LED strip, etc.)

Do NOT mention any brand names, logos, or model numbers.
Output only the description as a single dense paragraph — no headers, no lists."""


# ── Helper functies ───────────────────────────────────────────────────────────
def uploaded_file_to_part(uploaded_file):
    from google.genai import types
    raw = uploaded_file.read()
    mime = uploaded_file.type or "image/jpeg"
    return types.Part.from_bytes(data=raw, mime_type=mime)


def analyze_scene_image(client, scene_file):
    from google.genai import types
    scene_file.seek(0)
    parts = [
        uploaded_file_to_part(scene_file),
        types.Part.from_text(text=SCENE_ANALYSIS_PROMPT),
    ]
    for model in ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]:
        try:
            response = client.models.generate_content(model=model, contents=parts)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if any(code in err for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"]):
                continue
            raise
    raise RuntimeError("Alle modellen zijn momenteel overbelast. Probeer het over een minuut opnieuw.")


def analyze_product_images(client, uploaded_files, product_type):
    from google.genai import types

    parts = [uploaded_file_to_part(f) for f in uploaded_files]
    parts.append(
        types.Part.from_text(
            text=f"The product type is: {product_type}.\n\n{ANALYSIS_PROMPT}"
        )
    )

    for model in ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]:
        try:
            response = client.models.generate_content(model=model, contents=parts)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if any(code in err for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"]):
                continue
            raise
    raise RuntimeError("Alle modellen zijn momenteel overbelast. Probeer het over een minuut opnieuw.")


def build_final_prompt(product_description, product_type, scenario_text, lighting, mood, negative_prompt):
    return (
        f"Professional lifestyle product photograph. "
        f"A {product_type} — described as: {product_description} — "
        f"is {scenario_text} "
        f"Lighting style: {lighting}. "
        f"Visual mood: {mood}. "
        f"Photorealistic, high-resolution, commercial product photography. "
        f"Avoid: {negative_prompt}"
    )



def generate_with_pollinations(prompt):
    import requests, urllib.parse
    encoded = urllib.parse.quote(prompt[:800])
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=768&model=flux&nologo=true&enhance=false"
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Pollinations fout {response.status_code}: {response.text[:200]}")
    return response.content


def generate_lifestyle_image(client, prompt, use_imagen3=False):
    from google.genai import types

    if use_imagen3:
        response = client.models.generate_images(
            model="imagen-3.0-generate-001",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="4:3",
            ),
        )
        return response.generated_images[0].image.image_bytes
    else:
        errors = []
        for model in ["gemini-2.5-flash-image", "gemini-2.0-flash-exp", "gemini-2.0-flash-preview-image-generation"]:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )
                candidate = response.candidates[0]
                text_parts = []
                for part in candidate.content.parts:
                    if part.inline_data is not None:
                        return part.inline_data.data
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    raise RuntimeError(f"Model stuurde tekst: {' '.join(text_parts)[:300]}")
                finish = getattr(candidate, "finish_reason", None)
                raise RuntimeError(f"Geen afbeelding (finish_reason: {finish})")
            except RuntimeError:
                raise
            except Exception as e:
                errors.append(f"{model}: {e}")
                continue
        raise RuntimeError("Alle modellen gefaald:\n" + "\n".join(errors))


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📸 Lifestyle Image Generator")
st.caption(
    "Upload productfoto's → kies een lifestyle scène → genereer een fotorealistische "
    "marketingafbeelding met Google Imagen 3."
)
st.divider()

client = get_client()
if client is None:
    st.error(
        "**GEMINI_API_KEY niet gevonden.** "
        "Voeg de sleutel toe aan `.streamlit/secrets.toml` (lokaal) of via de "
        "Secrets-instellingen op Streamlit Community Cloud."
    )
    st.stop()

# ── Twee kolommen ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    # ── 1. Productconfiguratie ────────────────────────────────────────────────
    st.subheader("1  Productconfiguratie")

    product_type = st.selectbox(
        "Producttype",
        options=PRODUCT_TYPES,
        index=0,
        help="Selecteer het type product dat je fotografeert.",
    )

    uploaded_files = st.file_uploader(
        "Upload productfoto's (voor-, zij- en bovenaanzicht — max. 5)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload scherpe, goed belichte foto's van je product vanuit meerdere hoeken.",
    )

    if uploaded_files:
        cols = st.columns(min(len(uploaded_files), 3))
        for i, f in enumerate(uploaded_files[:3]):
            cols[i].image(f, caption=f.name, use_container_width=True)
        if len(uploaded_files) > 3:
            st.caption(f"+ nog {len(uploaded_files) - 3} afbeelding(en) geüpload.")

    st.divider()

    # ── 2. Merkrichtlijnen ────────────────────────────────────────────────────
    st.subheader("2  Merkrichtlijnen")

    with st.expander("Merkrichtlijnen bewerken", expanded=False):
        lighting = st.text_area(
            "Belichting & Kleuren",
            value=DEFAULT_LIGHTING,
            height=90,
            help="Beschrijft de gewenste lichtomstandigheden en kleurpalet (in het Engels voor beste resultaat).",
        )
        mood = st.text_area(
            "Visuele Stijl",
            value=DEFAULT_MOOD,
            height=90,
            help="Beschrijft de algehele sfeer en fotografiestijl (in het Engels voor beste resultaat).",
        )
        negative_prompt = st.text_area(
            "Negatieve Prompt (wat te vermijden)",
            value=DEFAULT_NEGATIVE,
            height=90,
            help="Alles wat NIET in het beeld mag verschijnen (in het Engels voor beste resultaat).",
        )

    st.divider()

    # ── 3. Scenariokeuze ──────────────────────────────────────────────────────
    st.subheader("3  Lifestyle Scenario")

    scenario_name = st.selectbox(
        "Kies een scenario",
        options=list(SCENARIOS.keys()),
        index=0,
    )

    if scenario_name == "✏️ Eigen scenario":
        custom_scenario = st.text_area(
            "Beschrijf je eigen scène",
            placeholder=(
                "Schrijf bij voorkeur in het Engels voor het beste resultaat.\n\n"
                "Voorbeeld: placed on a white marble kitchen counter next to a "
                "glass of orange juice and a folded linen napkin. Bright morning "
                "light streaming through a window."
            ),
            height=130,
            help="Beschrijf de omgeving, objecten naast het product, belichting en sfeer. "
                 "Engels geeft de scherpste resultaten.",
        )
        scenario_text = custom_scenario.strip()
        scene_image = None
        if scenario_text:
            st.info(f"**Jouw scène:** Het product is {scenario_text}")
        else:
            st.warning("Vul hierboven een scenariobeschrijving in om te kunnen genereren.")
    elif scenario_name == "📷 Voorbeeldafbeelding":
        scene_image = st.file_uploader(
            "Upload een voorbeeldafbeelding van de gewenste scène",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            help="Upload een foto van de omgeving of setting waar je het product in wilt plaatsen. "
                 "Gemini analyseert de scène automatisch.",
        )
        scenario_text = None
        if scene_image:
            st.image(scene_image, caption="Voorbeeldscène", use_container_width=True)
            st.info("Gemini analyseert deze scène automatisch bij het genereren.")
        else:
            st.warning("Upload een voorbeeldafbeelding om te kunnen genereren.")
    else:
        scenario_text = SCENARIOS[scenario_name]
        scene_image = None
        st.info(f"**Scène:** Het product is {scenario_text}")

    st.divider()

    # ── 4. Beeldgeneratiemodel ────────────────────────────────────────────────
    st.subheader("4  Beeldkwaliteit")

    image_model = st.radio(
        "Kies het beeldgeneratiemodel",
        options=[
            "FLUX — Pollinations.ai (gratis, testen)",
            "Imagen 3 — Google (~€0,03/afbeelding)",
            "Gemini Flash (experimenteel)",
        ],
        index=0,
        help="FLUX via Pollinations.ai is volledig gratis, geen API-sleutel nodig. Imagen 3 geeft de hoogste kwaliteit maar vereist Google Cloud billing.",
    )
    use_hf = image_model.startswith("FLUX")
    use_imagen3 = image_model.startswith("Imagen 3")

    if use_hf:
        st.info(
            "**FLUX via Pollinations.ai** — volledig gratis, geen API-sleutel nodig. "
            "Ideaal om de prompt-kwaliteit te testen voor productiegebruik."
        )
    elif use_imagen3:
        st.info(
            "**Imagen 3** — Google. Vereist actieve Google Cloud billing. "
            "Kosten: ~€0,03–0,04 per afbeelding."
        )
    else:
        st.warning("**Gemini Flash** is experimenteel en mogelijk beperkt beschikbaar.")

    st.divider()

    # ── 5. Waarschuwing gratis limiet ─────────────────────────────────────────
    st.warning(
        "**Let op: API-limieten — lees dit voordat je genereert.**\n\n"
        "- Gemini Flash analyse: ~10 verzoeken per minuut op de gratis laag.\n"
        "- Snel achter elkaar genereren kan een tijdelijke blokkade veroorzaken.\n\n"
        "Controleer je verbruik via **Google AI Studio → API usage**."
    )

    confirmed = st.checkbox(
        "Ik begrijp de API-limieten en zal niet overmatig genereren."
    )

    custom_empty = (scenario_name == "✏️ Eigen scenario" and not scenario_text)
    scene_empty = (scenario_name == "📷 Voorbeeldafbeelding" and not scene_image)

    generate_btn = st.button(
        "Lifestyle afbeelding genereren",
        type="primary",
        disabled=not confirmed or not uploaded_files or custom_empty or scene_empty,
        use_container_width=True,
    )

    if not uploaded_files:
        st.caption("Upload minimaal één productafbeelding om te beginnen.")
    elif custom_empty:
        st.caption("Vul een scenariobeschrijving in om te kunnen genereren.")
    elif scene_empty:
        st.caption("Upload een voorbeeldafbeelding van de scène om te kunnen genereren.")
    elif not confirmed:
        st.caption("Vink het bevestigingsvakje hierboven aan om de knop te activeren.")

# ── Rechterkolom: uitvoer ─────────────────────────────────────────────────────
with right:
    st.subheader("4  Gegenereerde Afbeelding")

    if "product_description" not in st.session_state:
        st.session_state.product_description = None
    if "last_files" not in st.session_state:
        st.session_state.last_files = []
    if "scene_description" not in st.session_state:
        st.session_state.scene_description = None
    if "last_scene_file" not in st.session_state:
        st.session_state.last_scene_file = None

    if generate_btn:
        current_file_names = [f.name for f in uploaded_files]
        files_changed = current_file_names != st.session_state.last_files

        if files_changed or st.session_state.product_description is None:
            with st.spinner("Stap 1/3 — Productafbeeldingen analyseren met Gemini..."):
                try:
                    for f in uploaded_files:
                        f.seek(0)
                    description = analyze_product_images(client, uploaded_files, product_type)
                    st.session_state.product_description = description
                    st.session_state.last_files = current_file_names
                except Exception as e:
                    st.error(f"Afbeeldingsanalyse mislukt: {e}")
                    st.stop()
        else:
            description = st.session_state.product_description

        if scenario_name == "📷 Voorbeeldafbeelding":
            scene_file_name = scene_image.name if scene_image else None
            if scene_file_name != st.session_state.last_scene_file or st.session_state.scene_description is None:
                with st.spinner("Stap 2/3 — Voorbeeldscène analyseren met Gemini..."):
                    try:
                        scene_desc = analyze_scene_image(client, scene_image)
                        st.session_state.scene_description = scene_desc
                        st.session_state.last_scene_file = scene_file_name
                    except Exception as e:
                        st.error(f"Scèneanalyse mislukt: {e}")
                        st.stop()
            active_scenario_text = st.session_state.scene_description
            with st.expander("Scènebeschrijving (automatisch gegenereerd)", expanded=False):
                st.write(active_scenario_text)
            step_label = "Stap 3/3"
        else:
            active_scenario_text = scenario_text
            step_label = "Stap 2/2"

        with st.expander("Productbeschrijving (automatisch gegenereerd)", expanded=False):
            st.write(st.session_state.product_description)

        final_prompt = build_final_prompt(
            st.session_state.product_description,
            product_type,
            active_scenario_text,
            lighting,
            mood,
            negative_prompt,
        )

        if use_hf:
            model_label = "FLUX (Pollinations.ai)"
        elif use_imagen3:
            model_label = "Imagen 3 (Google)"
        else:
            model_label = "Gemini Flash"

        with st.expander(f"Volledige prompt naar {model_label}", expanded=False):
            st.code(final_prompt, language=None)

        with st.spinner(f"{step_label} — Lifestyle afbeelding genereren met {model_label}..."):
            try:
                if use_hf:
                    image_bytes = generate_with_pollinations(final_prompt)
                elif use_imagen3:
                    image_bytes = generate_lifestyle_image(get_imagen_client(), final_prompt, use_imagen3=True)
                else:
                    image_bytes = generate_lifestyle_image(get_flash_image_client(), final_prompt, use_imagen3=False)
            except Exception as e:
                st.error(f"Afbeeldingsgeneratie mislukt: {e}")
                st.stop()

        st.success("Afbeelding succesvol gegenereerd!")
        st.image(image_bytes, use_container_width=True)
        safe_name = (scenario_name.lower().replace(" ", "_").replace("/", "")
                     .replace("✏️_", "eigen_").replace("📷_", "scene_"))
        st.download_button(
            label="Afbeelding downloaden",
            data=image_bytes,
            file_name=f"lifestyle_{safe_name}.png",
            mime="image/png",
            use_container_width=True,
        )

    else:
        st.markdown(
            """
            <div style="
                border: 2px dashed #ccc;
                border-radius: 12px;
                height: 380px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #888;
                font-size: 1.1rem;
            ">
                Jouw gegenereerde afbeelding verschijnt hier
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.product_description:
            with st.expander("Gecachede productbeschrijving (vorige sessie)", expanded=False):
                st.write(st.session_state.product_description)
