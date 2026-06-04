import streamlit as st
import base64
import io
from PIL import Image
from datetime import datetime

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
    "Kabel",
    "Oplader",
    "Draadloze Oordopjes",
    "Smartphonehoesje",
    "Laptophoes",
    "Smartwatch",
    "Draadloze Speaker",
    "Telefoonhouder",
    "Anders",
]

SCENARIOS = {
    "Treinreis": (
        "placed flat on a textured plastic train tray-table next to a dutch passport "
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

TELEFOONHOUDER_SCENARIOS = {
    "De Dagelijkse Forens": (
        "in a car interior during a morning commute. A premium smartphone on the mount displays "
        "a GPS navigation map screen. Crisp, bright morning sunlight streams through the front "
        "windshield, casting realistic soft shadows across the clean car interior, with the "
        "steering wheel softly blurred in the background."
    ),
    "Nachtelijke Stadsrit": (
        "in a car interior at night. A smartphone on the mount displays a dark-mode music "
        "playlist. The cabin is dimly lit with subtle ambient dashboard lights, while the side "
        "window reveals a cinematic, softly blurred bokeh of colorful city neon lights and "
        "traffic at night."
    ),
    "De Zomerse Roadtrip": (
        "in a car interior at golden-hour sunset. A thin black USB-C cable runs neatly from the "
        "bottom of the mount. Warm, golden sunset light floods the car interior, creating long "
        "soft shadows, with an open scenic highway stretching into the distance softly blurred "
        "through the windshield."
    ),
    "Premium & Luxe": (
        "in the interior of a luxury electric vehicle with minimalist carbon-fiber dashboard "
        "trim. A smartphone is attached horizontally to the mount. The lighting is clean, "
        "diffused, and overhead, highlighting the premium matte textures and metallic joints of "
        "the mount — pristine, high-tech, and perfectly proportional to the high-end interior."
    ),
    "✏️ Eigen scenario": None,
    "📷 Voorbeeldafbeelding": None,
}

MOUNT_TYPES = {
    "Ventilatierooster": "clipped onto the car air vent via an air vent clip mount",
    "Zuignap — Dashboard": "attached to the car dashboard via a suction cup base",
    "Zuignap — Voorruit": "mounted on the inside of the front windshield via a suction cup arm",
    "Bekerhouder": "inserted and secured in the car cup holder",
    "CD-sleuf": "locked into the car's CD slot via a CD slot insert adapter",
    "Hoofdsteun (achterbank)": "strapped to the rear headrest, facing the back seat passengers",
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
    "trash, chaotic backgrounds, cheap-looking plastic, low-quality, CGI look, "
    "plastic texture, watermark, blurry product."
)

ASPECT_OPTIONS = {
    "4:3 (Standaard)": "4:3",
    "1:1 (Instagram vierkant)": "1:1",
    "9:16 (Story / Portrait)": "9:16",
    "16:9 (Webbanner / Landscape)": "16:9",
}

POLLINATIONS_SIZES = {
    "4:3": (1024, 768),
    "1:1": (1024, 1024),
    "9:16": (576, 1024),
    "16:9": (1024, 576),
}

# Per product type: how the product physically appears in any scene.
# Appended to every scenario so the AI knows the correct depiction regardless of scene choice.
PRODUCT_SCENE_MODIFIER = {
    "Kabel": (
        "The cable lies naturally in the scene with both connector ends spread apart and "
        "clearly identifiable. The connectors specified above are the absolute top priority — "
        "they must be rendered with exact accuracy."
    ),
    "Powerbank": (
        "The powerbank is displayed with its ports and any LED indicators facing the viewer "
        "and clearly visible."
    ),
    "Oplader": (
        "The charger is positioned with its port(s) and charging surface facing the viewer "
        "and clearly visible."
    ),
    "Smartphonehoesje": (
        "The case is fitted snugly on a matching smartphone, showcasing its design, texture, "
        "and color."
    ),
}

SCENE_ANALYSIS_PROMPT = """You are a professional product photographer's assistant analyzing a reference scene image.
Describe the full scene so it can be recreated as a lifestyle product photograph.

Include ALL of the following if visible:
- The surface or foreground where a product could be placed (material, texture, color)
- Nearby props and objects in the scene
- Background environment and depth
- Lighting conditions (direction, quality, warm/cool, soft/hard)
- Overall atmosphere and mood
- Any people present: approximate age range, gender expression, clothing style, pose,
  and what they are doing — but do NOT describe or reference their face or identity

Output a single concise paragraph starting with "placed on/in/near..." that describes
where and how a product would appear in this scene, including any human lifestyle context.
No headers, no lists."""

ANALYSIS_PROMPT = """You are a professional product photographer's assistant.
Analyze the uploaded product image(s) and produce a hyper-detailed, technical
visual description of the physical product, including all branding exactly as it appears.

Include ALL of the following if visible:
- Overall shape and form factor (dimensions, thickness, curvature)
- Surface textures (matte, glossy, brushed metal, rubberized, etc.)
- Color(s) — use precise color names (e.g. "slate grey", "cream white")
- Visible ports, buttons, indicators, seams, and their placement
- Material quality cues (premium metal, soft-touch plastic, fabric, etc.)
- Any distinguishing physical features (rounded corners, ridges, LED strip, etc.)
- Brand name, logo placement, typography, and any text printed on the product

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


def build_final_prompt(product_description, product_type, scenario_text, lighting, mood,
                       negative_prompt, mount_type=None, product_specs=None):
    scene_modifier = PRODUCT_SCENE_MODIFIER.get(product_type, "")
    modifier_clause = f" {scene_modifier}" if scene_modifier else ""

    if mount_type:
        placement = f"is {mount_type}. The holder {scenario_text}"
    else:
        placement = f"is {scenario_text}"

    # For technical products: put specs at the very top so the AI treats them as highest priority.
    if product_type in ("Kabel", "Powerbank", "Oplader") and product_specs:
        return (
            f"{product_specs} "
            f"Professional lifestyle product photograph. "
            f"A {product_type} — described as: {product_description}. "
            f"The product {placement}{modifier_clause} "
            f"Lighting style: {lighting}. "
            f"Visual mood: {mood}. "
            f"Photorealistic, high-resolution, commercial product photography. "
            f"Avoid: {negative_prompt}"
        )

    specs_clause = f" {product_specs}" if product_specs else ""
    return (
        f"Professional lifestyle product photograph. "
        f"A {product_type} — described as: {product_description}.{specs_clause} "
        f"The product {placement}{modifier_clause} "
        f"Lighting style: {lighting}. "
        f"Visual mood: {mood}. "
        f"Photorealistic, high-resolution, commercial product photography. "
        f"Avoid: {negative_prompt}"
    )


def generate_with_pollinations(prompt, aspect_ratio="4:3", number_of_images=1):
    import requests, urllib.parse, random
    w, h = POLLINATIONS_SIZES.get(aspect_ratio, (1024, 768))
    encoded = urllib.parse.quote(prompt[:800])
    images = []
    for _ in range(number_of_images):
        seed = random.randint(0, 999999)
        url = (f"https://image.pollinations.ai/prompt/{encoded}"
               f"?width={w}&height={h}&model=flux&nologo=true&enhance=false&seed={seed}")
        response = requests.get(url, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"Pollinations fout {response.status_code}: {response.text[:200]}")
        images.append(response.content)
    return images


def generate_lifestyle_image(client, prompt, use_imagen3=False, number_of_images=1, aspect_ratio="4:3"):
    from google.genai import types

    if use_imagen3:
        response = client.models.generate_images(
            model="imagen-3.0-generate-001",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=number_of_images,
                aspect_ratio=aspect_ratio,
            ),
        )
        return [img.image_bytes for img in response.generated_images]
    else:
        errors = []
        for model in ["gemini-2.5-flash-image", "gemini-2.0-flash-exp", "gemini-2.0-flash-preview-image-generation"]:
            try:
                images = []
                for _ in range(number_of_images):
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
                            images.append(part.inline_data.data)
                            break
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)
                    else:
                        if text_parts:
                            raise RuntimeError(f"Model stuurde tekst: {' '.join(text_parts)[:300]}")
                        finish = getattr(candidate, "finish_reason", None)
                        raise RuntimeError(f"Geen afbeelding (finish_reason: {finish})")
                return images
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
    "marketingafbeelding."
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

# ── Session state initialisatie ───────────────────────────────────────────────
st.session_state.setdefault("product_description", None)
st.session_state.setdefault("last_files", [])
st.session_state.setdefault("scene_description", None)
st.session_state.setdefault("last_scene_file", None)
st.session_state.setdefault("description_edit", "")
st.session_state.setdefault("history", [])
st.session_state.setdefault("brand_profiles", {
    "Standaard": {"lighting": DEFAULT_LIGHTING, "mood": DEFAULT_MOOD, "negative": DEFAULT_NEGATIVE}
})
st.session_state.setdefault("last_final_prompt", None)
st.session_state.setdefault("last_gen_cfg", None)
st.session_state.setdefault("lighting", DEFAULT_LIGHTING)
st.session_state.setdefault("mood", DEFAULT_MOOD)
st.session_state.setdefault("negative_prompt", DEFAULT_NEGATIVE)

# Apply any pending description update BEFORE widgets are instantiated
if "_pending_description" in st.session_state:
    st.session_state.description_edit = st.session_state.pop("_pending_description")

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

    if product_type == "Telefoonhouder":
        mount_label = st.selectbox(
            "Bevestigingstype",
            options=list(MOUNT_TYPES.keys()),
            help="Hoe wordt de houder in de auto bevestigd?",
        )
        mount_type = MOUNT_TYPES[mount_label]
        magsafe_holder = st.checkbox("MagSafe compatibel (magneetring zichtbaar op het montageoppervlak)")
        product_specs = "The mount face has a visible MagSafe magnetic ring." if magsafe_holder else None

    elif product_type == "Kabel":
        mount_type = None
        col_a, col_b = st.columns(2)
        connector_a = col_a.selectbox(
            "Connector kant 1",
            ["USB-C", "USB-A", "Lightning", "Micro-USB"],
            help="Stekkertype aan de ene kant van de kabel.",
        )
        connector_b = col_b.selectbox(
            "Connector kant 2",
            ["USB-C", "USB-A", "Lightning", "Micro-USB"],
            help="Stekkertype aan de andere kant van de kabel.",
        )
        product_specs = (
            f"The cable has a {connector_a} connector on one end and a {connector_b} connector "
            f"on the other. Both connectors must be rendered accurately and clearly visible."
        )

    elif product_type == "Powerbank":
        mount_type = None
        ports = st.multiselect(
            "Aanwezige poorten",
            ["USB-C", "USB-A", "Micro-USB", "Lightning (output)"],
            help="Selecteer alle poorten die op de powerbank aanwezig zijn.",
        )
        magsafe_pb = st.checkbox("MagSafe / Qi2 wireless charging (ring zichtbaar op de achterkant)")
        specs = []
        if ports:
            specs.append(f"visible ports on the housing: {', '.join(ports)}")
        if magsafe_pb:
            specs.append("MagSafe / Qi2 magnetic wireless charging ring clearly visible on the back surface")
        product_specs = "Product specifications — " + "; ".join(specs) + "." if specs else None

    elif product_type == "Oplader":
        mount_type = None
        ports = st.multiselect(
            "Aanwezige poorten",
            ["USB-C", "USB-A"],
            help="Selecteer alle poorten die op de oplader aanwezig zijn.",
        )
        magsafe_ch = st.checkbox("MagSafe / Qi wireless (voor draadloze pads of stands)")
        gan = st.checkbox("GaN technologie (compact formaat, geen groot blokje)")
        specs = []
        if ports:
            specs.append(f"visible ports: {', '.join(ports)}")
        if magsafe_ch:
            specs.append("MagSafe / Qi wireless charging surface clearly visible")
        if gan:
            specs.append("GaN compact design — small form factor, no large brick shape")
        product_specs = "Product specifications — " + "; ".join(specs) + "." if specs else None

    else:
        mount_type = None
        product_specs = None

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

    with st.expander("Productbeschrijving bewerken", expanded=False):
        st.text_area(
            "Productbeschrijving",
            key="description_edit",
            height=120,
            help="Wordt automatisch ingevuld na de eerste generatie. Pas aan indien Gemini iets verkeerd beschrijft.",
            placeholder="Wordt automatisch ingevuld na eerste generatie. Pas hier aan indien gewenst.",
            label_visibility="collapsed",
        )

    st.divider()

    # ── 2. Merkrichtlijnen ────────────────────────────────────────────────────
    st.subheader("2  Merkrichtlijnen")

    with st.expander("Merkrichtlijnen bewerken", expanded=False):
        # Profielbeheer
        profiel_namen = ["— Kies profiel —"] + list(st.session_state.brand_profiles.keys())
        pcol1, pcol2 = st.columns([3, 1])
        gekozen = pcol1.selectbox("Laad profiel", profiel_namen, key="profiel_keuze", label_visibility="collapsed")
        if pcol2.button("Laden", use_container_width=True) and gekozen != "— Kies profiel —":
            p = st.session_state.brand_profiles[gekozen]
            st.session_state.lighting = p["lighting"]
            st.session_state.mood = p["mood"]
            st.session_state.negative_prompt = p["negative"]
            st.rerun()

        st.divider()

        lighting = st.text_area(
            "Belichting & Kleuren",
            key="lighting",
            height=90,
            help="Beschrijft de gewenste lichtomstandigheden en kleurpalet (in het Engels voor beste resultaat).",
        )
        mood = st.text_area(
            "Visuele Stijl",
            key="mood",
            height=90,
            help="Beschrijft de algehele sfeer en fotografiestijl (in het Engels voor beste resultaat).",
        )
        negative_prompt = st.text_area(
            "Negatieve Prompt (wat te vermijden)",
            key="negative_prompt",
            height=90,
            help="Alles wat NIET in het beeld mag verschijnen (in het Engels voor beste resultaat).",
        )

        st.divider()

        # Profiel opslaan
        scol1, scol2 = st.columns([3, 1])
        profiel_naam = scol1.text_input(
            "Naam", placeholder="bijv. Zomercollectie", label_visibility="collapsed", key="nieuw_profiel_naam"
        )
        if scol2.button("Opslaan", use_container_width=True, key="opslaan_profiel"):
            if profiel_naam.strip():
                st.session_state.brand_profiles[profiel_naam.strip()] = {
                    "lighting": st.session_state.lighting,
                    "mood": st.session_state.mood,
                    "negative": st.session_state.negative_prompt,
                }
                st.success(f"Profiel '{profiel_naam.strip()}' opgeslagen.")
            else:
                st.warning("Vul een profielnaam in om op te slaan.")

    st.divider()

    # ── 3. Scenariokeuze ──────────────────────────────────────────────────────
    st.subheader("3  Lifestyle Scenario")

    active_scenarios = TELEFOONHOUDER_SCENARIOS if product_type == "Telefoonhouder" else SCENARIOS

    scenario_name = st.selectbox(
        "Kies een scenario",
        options=list(active_scenarios.keys()),
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
            help="Beschrijf de omgeving, objecten naast het product, belichting en sfeer.",
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
            help="Upload een foto van de omgeving of setting waar je het product in wilt plaatsen.",
        )
        scenario_text = None
        if scene_image:
            st.image(scene_image, caption="Voorbeeldscène", use_container_width=True)
            st.info("Gemini analyseert deze scène automatisch bij het genereren.")
        else:
            st.warning("Upload een voorbeeldafbeelding om te kunnen genereren.")
    else:
        scenario_text = active_scenarios[scenario_name]
        scene_image = None
        st.info(f"**Scène:** Het product is {scenario_text}")

    st.divider()

    # ── 4. Beeldkwaliteit ─────────────────────────────────────────────────────
    st.subheader("4  Beeldkwaliteit")

    image_model = st.radio(
        "Beeldgeneratiemodel",
        options=[
            "FLUX — Pollinations.ai (gratis, testen)",
            "Imagen 3 — Google (~€0,03/afbeelding)",
            "Gemini Flash (experimenteel)",
        ],
        index=0,
        help="FLUX via Pollinations.ai is volledig gratis. Imagen 3 geeft de hoogste kwaliteit maar vereist Google Cloud billing.",
    )
    use_hf = image_model.startswith("FLUX")
    use_imagen3 = image_model.startswith("Imagen 3")

    if use_hf:
        st.info("**FLUX via Pollinations.ai** — volledig gratis, geen API-sleutel nodig.")
    elif use_imagen3:
        st.info("**Imagen 3** — Google. Vereist actieve Google Cloud billing. ~€0,03–0,04/afbeelding.")
    else:
        st.warning("**Gemini Flash** is experimenteel en mogelijk beperkt beschikbaar.")

    aspect_label = st.selectbox(
        "Aspectratio",
        options=list(ASPECT_OPTIONS.keys()),
        index=0,
        help="Kies het formaat voor de gegenereerde afbeelding.",
    )
    aspect_ratio_key = ASPECT_OPTIONS[aspect_label]

    number_of_images = st.slider(
        "Aantal varianten",
        min_value=1,
        max_value=3,
        value=1,
        help="Genereer meerdere varianten tegelijk. Let op: elke variant verbruikt API-quota.",
    )

    st.divider()

    # ── 5. Bevestiging & genereren ────────────────────────────────────────────
    st.warning(
        "**Let op: API-limieten — lees dit voordat je genereert.**\n\n"
        "- Gemini Flash analyse: ~10 verzoeken per minuut op de gratis laag.\n"
        "- Snel achter elkaar genereren kan een tijdelijke blokkade veroorzaken.\n\n"
        "Controleer je verbruik via **Google AI Studio → API usage**."
    )

    confirmed = st.checkbox("Ik begrijp de API-limieten en zal niet overmatig genereren.")

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
    st.subheader("Uitvoer")

    is_regen = st.session_state.pop("regenerate", False)

    if generate_btn or is_regen:
        safe_name = (scenario_name.lower().replace(" ", "_").replace("/", "")
                     .replace("✏️_", "eigen_").replace("📷_", "scene_"))

        if is_regen and st.session_state.last_final_prompt and st.session_state.last_gen_cfg:
            # ── Opnieuw genereren: sla analyse over ───────────────────────────
            final_prompt = st.session_state.last_final_prompt
            cfg = st.session_state.last_gen_cfg
            r_use_hf = cfg["use_hf"]
            r_use_imagen3 = cfg["use_imagen3"]
            r_aspect = cfg["aspect_ratio"]
            r_n = cfg["n"]
            model_label = cfg["model_label"]
            safe_name = cfg.get("safe_name", safe_name)
            step_label = "Opnieuw genereren"
        else:
            # ── Normale flow ──────────────────────────────────────────────────
            current_file_names = [f.name for f in uploaded_files]
            files_changed = current_file_names != st.session_state.last_files

            if files_changed or st.session_state.product_description is None:
                with st.spinner("Stap 1 — Productafbeeldingen analyseren met Gemini..."):
                    try:
                        for f in uploaded_files:
                            f.seek(0)
                        description = analyze_product_images(client, uploaded_files, product_type)
                        st.session_state.product_description = description
                        st.session_state.last_files = current_file_names
                        if files_changed or not st.session_state.description_edit:
                            st.session_state["_pending_description"] = description
                    except Exception as e:
                        st.error(f"Afbeeldingsanalyse mislukt: {e}")
                        st.stop()

            if scenario_name == "📷 Voorbeeldafbeelding":
                scene_file_name = scene_image.name if scene_image else None
                if scene_file_name != st.session_state.last_scene_file or st.session_state.scene_description is None:
                    with st.spinner("Stap 2 — Voorbeeldscène analyseren met Gemini..."):
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
                step_label = "Stap 3"
            else:
                active_scenario_text = scenario_text
                step_label = "Stap 2"

            with st.expander("Productbeschrijving (gebruikt voor generatie)", expanded=False):
                st.write(st.session_state.description_edit or st.session_state.product_description)

            final_prompt = build_final_prompt(
                st.session_state.description_edit or st.session_state.product_description,
                product_type,
                active_scenario_text,
                lighting,
                mood,
                negative_prompt,
                mount_type=mount_type,
                product_specs=product_specs,
            )

            if use_hf:
                model_label = "FLUX (Pollinations.ai)"
            elif use_imagen3:
                model_label = "Imagen 3 (Google)"
            else:
                model_label = "Gemini Flash"

            r_use_hf, r_use_imagen3 = use_hf, use_imagen3
            r_aspect, r_n = aspect_ratio_key, number_of_images

            st.session_state.last_final_prompt = final_prompt
            st.session_state.last_gen_cfg = {
                "use_hf": use_hf, "use_imagen3": use_imagen3,
                "aspect_ratio": aspect_ratio_key, "n": number_of_images,
                "model_label": model_label, "safe_name": safe_name,
                "product_specs": product_specs,
            }

        with st.expander(f"Volledige prompt naar {model_label}", expanded=False):
            st.code(final_prompt, language=None)

        variant_label = f"{r_n} variant{'en' if r_n > 1 else ''}"
        with st.spinner(f"{step_label} — {variant_label} genereren met {model_label}..."):
            try:
                if r_use_hf:
                    images = generate_with_pollinations(final_prompt, r_aspect, r_n)
                elif r_use_imagen3:
                    images = generate_lifestyle_image(get_imagen_client(), final_prompt, True, r_n, r_aspect)
                else:
                    images = generate_lifestyle_image(get_flash_image_client(), final_prompt, False, r_n, r_aspect)
            except Exception as e:
                st.error(f"Afbeeldingsgeneratie mislukt: {e}")
                st.stop()

        st.success(f"{'Afbeelding' if len(images) == 1 else f'{len(images)} afbeeldingen'} succesvol gegenereerd!")

        if len(images) == 1:
            st.image(images[0], use_container_width=True)
            st.download_button(
                label="Afbeelding downloaden",
                data=images[0],
                file_name=f"lifestyle_{safe_name}.png",
                mime="image/png",
                use_container_width=True,
            )
        else:
            img_cols = st.columns(len(images))
            for i, (col, img) in enumerate(zip(img_cols, images)):
                col.image(img, use_container_width=True)
                col.download_button(
                    f"Download {i + 1}",
                    data=img,
                    file_name=f"lifestyle_{safe_name}_{i + 1}.png",
                    mime="image/png",
                    use_container_width=True,
                    key=f"dl_{i}",
                )

        if st.button("🔄 Opnieuw genereren", use_container_width=True):
            st.session_state.regenerate = True
            st.rerun()

        # Opslaan in sessiegeschiedenis
        st.session_state.history.insert(0, {
            "images": images,
            "scenario": scenario_name,
            "model": model_label,
            "timestamp": datetime.now().strftime("%H:%M"),
        })
        st.session_state.history = st.session_state.history[:10]

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

    # ── Sessiegeschiedenis ────────────────────────────────────────────────────
    if st.session_state.history:
        st.divider()
        with st.expander(f"Sessiegeschiedenis ({len(st.session_state.history)} generaties)", expanded=False):
            for idx, item in enumerate(st.session_state.history):
                st.caption(f"**{item['timestamp']}** — {item['scenario']} — {item['model']}")
                h_cols = st.columns(min(len(item["images"]), 3))
                for j, (col, img) in enumerate(zip(h_cols, item["images"])):
                    col.image(img, use_container_width=True)
                    col.download_button(
                        "⬇",
                        data=img,
                        file_name=f"history_{idx}_{j}.png",
                        mime="image/png",
                        key=f"hist_{idx}_{j}",
                        use_container_width=True,
                    )
                if idx < len(st.session_state.history) - 1:
                    st.divider()
