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

# ── Gemini client ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    try:
        from google import genai
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
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


def generate_lifestyle_image(client, prompt):
    from google.genai import types

    response = client.models.generate_images(
        model="imagen-3.0-generate-001",
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="4:3",
        ),
    )
    return response.generated_images[0].image.image_bytes


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
        if scenario_text:
            st.info(f"**Jouw scène:** Het product is {scenario_text}")
        else:
            st.warning("Vul hierboven een scenariobeschrijving in om te kunnen genereren.")
    else:
        scenario_text = SCENARIOS[scenario_name]
        st.info(f"**Scène:** Het product is {scenario_text}")

    st.divider()

    # ── 4. Waarschuwing gratis limiet ─────────────────────────────────────────
    st.warning(
        "**Let op: gratis API-limieten — lees dit voordat je genereert.**\n\n"
        "- Gemini Flash: ~10 verzoeken per minuut op de gratis laag.\n"
        "- Imagen 3: beperkt gratis quota — elke generatie verbruikt quota.\n"
        "- Snel achter elkaar genereren kan je gratis tegoed uitputten of "
        "een tijdelijke blokkade veroorzaken.\n\n"
        "Controleer je verbruik via **Google AI Studio → API usage**."
    )

    confirmed = st.checkbox(
        "Ik begrijp de API-limieten en zal niet overmatig genereren."
    )

    custom_empty = (scenario_name == "✏️ Eigen scenario" and not scenario_text)

    generate_btn = st.button(
        "Lifestyle afbeelding genereren",
        type="primary",
        disabled=not confirmed or not uploaded_files or custom_empty,
        use_container_width=True,
    )

    if not uploaded_files:
        st.caption("Upload minimaal één productafbeelding om te beginnen.")
    elif custom_empty:
        st.caption("Vul een scenariobeschrijving in om te kunnen genereren.")
    elif not confirmed:
        st.caption("Vink het bevestigingsvakje hierboven aan om de knop te activeren.")

# ── Rechterkolom: uitvoer ─────────────────────────────────────────────────────
with right:
    st.subheader("4  Gegenereerde Afbeelding")

    if "product_description" not in st.session_state:
        st.session_state.product_description = None
    if "last_files" not in st.session_state:
        st.session_state.last_files = []

    if generate_btn:
        current_file_names = [f.name for f in uploaded_files]
        files_changed = current_file_names != st.session_state.last_files

        if files_changed or st.session_state.product_description is None:
            with st.spinner("Stap 1/2 — Productafbeeldingen analyseren met Gemini..."):
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

        with st.expander("Productbeschrijving (automatisch gegenereerd)", expanded=False):
            st.write(st.session_state.product_description)

        final_prompt = build_final_prompt(
            st.session_state.product_description,
            product_type,
            scenario_text,
            lighting,
            mood,
            negative_prompt,
        )

        with st.expander("Volledige prompt naar Imagen 3", expanded=False):
            st.code(final_prompt, language=None)

        with st.spinner("Stap 2/2 — Lifestyle afbeelding genereren met Imagen 3..."):
            try:
                image_bytes = generate_lifestyle_image(client, final_prompt)
            except Exception as e:
                st.error(f"Afbeeldingsgeneratie mislukt: {e}")
                st.stop()

        st.success("Afbeelding succesvol gegenereerd!")
        st.image(image_bytes, use_container_width=True)
        safe_name = scenario_name.lower().replace(" ", "_").replace("/", "").replace("✏️_", "eigen_")
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
