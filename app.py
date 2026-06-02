import streamlit as st
import base64
import io
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lifestyle Image Generator",
    page_icon="📸",
    layout="wide",
)

# ── Gemini client (lazy-init so missing key shows a friendly error) ───────────
@st.cache_resource
def get_client():
    try:
        from google import genai
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except KeyError:
        return None

# ── Constants ─────────────────────────────────────────────────────────────────
PRODUCT_TYPES = [
    "Powerbank",
    "Wireless Earbuds",
    "Smartphone Case",
    "Laptop Sleeve",
    "Smart Watch",
    "Portable Speaker",
    "Other",
]

SCENARIOS = {
    "Train Travel": (
        "placed flat on a textured plastic train tray-table next to a dark blue passport "
        "and a ceramic coffee mug. Natural window light casting soft shadows, blurred "
        "landscape rushing by in the background."
    ),
    "Music Festival": (
        "placed on a weathered wooden picnic table at a bustling outdoor music festival, "
        "next to a colorful festival wristband and trendy sunglasses. Casual lighting, "
        "blurred background of festival crowds and tents."
    ),
    "City Park": (
        "placed on a classic checkered picnic blanket in the lush green grass of a city "
        "park, next to a pair of sunglasses and a glass soda bottle. Warm, golden sunlight "
        "filtering through tree leaves."
    ),
    "Modern Office": (
        "placed on a real oak wood desk next to a modern wireless computer mouse and a "
        "black ink pen. Clean, everyday office environment with realistic proportions and "
        "reflections."
    ),
    "Nature/Outdoor": (
        "placed on a rough, dusty grey rock next to a folded paper map and a standard "
        "metallic climbing carabiner. Diffused, overcast daylight in a pine forest."
    ),
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


# ── Helper functions ──────────────────────────────────────────────────────────
def uploaded_file_to_part(uploaded_file):
    """Convert a Streamlit UploadedFile to a google-genai Part."""
    from google.genai import types
    raw = uploaded_file.read()
    mime = uploaded_file.type or "image/jpeg"
    return types.Part.from_bytes(data=raw, mime_type=mime)


def analyze_product_images(client, uploaded_files, product_type):
    """Step 1 — use Gemini 2.5 Flash to describe the product from the images."""
    from google.genai import types

    parts = [uploaded_file_to_part(f) for f in uploaded_files]
    parts.append(
        types.Part.from_text(
            text=f"The product type is: {product_type}.\n\n{ANALYSIS_PROMPT}"
        )
    )

    for model in ["gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            response = client.models.generate_content(model=model, contents=parts)
            return response.text.strip()
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                continue
            raise
    raise RuntimeError("All models unavailable. Please try again in a minute.")


def build_final_prompt(product_description, product_type, scenario_text, lighting, mood):
    return (
        f"Professional lifestyle product photograph. "
        f"A {product_type} — described as: {product_description} — "
        f"is {scenario_text} "
        f"Lighting style: {lighting}. "
        f"Visual mood: {mood}. "
        f"Photorealistic, high-resolution, commercial product photography."
    )


def generate_lifestyle_image(client, prompt, negative_prompt):
    """Step 2 — generate the lifestyle image with Imagen 3."""
    from google.genai import types

    response = client.models.generate_images(
        model="imagen-3.0-generate-001",
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="4:3",
            negative_prompt=negative_prompt,
        ),
    )
    return response.generated_images[0].image.image_bytes


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📸 Lifestyle Image Generator")
st.caption(
    "Upload your product photos → pick a lifestyle scene → generate a photorealistic "
    "marketing image powered by Google Imagen 3."
)
st.divider()

client = get_client()
if client is None:
    st.error(
        "**GEMINI_API_KEY not found.** "
        "Add it to `.streamlit/secrets.toml` locally, or to your app's Secrets on "
        "Streamlit Community Cloud."
    )
    st.stop()

# ── Layout: two columns ───────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    # ── 1. Product Configuration ──────────────────────────────────────────────
    st.subheader("1  Product Configuration")

    product_type = st.selectbox(
        "Product Type",
        options=PRODUCT_TYPES,
        index=0,
        help="Select the type of product you are photographing.",
    )

    uploaded_files = st.file_uploader(
        "Upload Product Images (Front, Side, Top — up to 5)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload clear, well-lit photos of your product from multiple angles.",
    )

    if uploaded_files:
        cols = st.columns(min(len(uploaded_files), 3))
        for i, f in enumerate(uploaded_files[:3]):
            cols[i].image(f, caption=f.name, use_container_width=True)
        if len(uploaded_files) > 3:
            st.caption(f"+ {len(uploaded_files) - 3} more image(s) uploaded.")

    st.divider()

    # ── 2. Brand Guidelines ───────────────────────────────────────────────────
    st.subheader("2  Brand Guidelines")

    with st.expander("Edit Brand Guidelines", expanded=False):
        lighting = st.text_area(
            "Lighting & Colors",
            value=DEFAULT_LIGHTING,
            height=90,
        )
        mood = st.text_area(
            "Visual Mood",
            value=DEFAULT_MOOD,
            height=90,
        )
        negative_prompt = st.text_area(
            "Negative Prompt (what to avoid)",
            value=DEFAULT_NEGATIVE,
            height=90,
        )
    # Show compact preview when collapsed
    if "lighting" not in dir():
        lighting = DEFAULT_LIGHTING
        mood = DEFAULT_MOOD
        negative_prompt = DEFAULT_NEGATIVE

    st.divider()

    # ── 3. Scenario Selection ─────────────────────────────────────────────────
    st.subheader("3  Lifestyle Scenario")

    scenario_name = st.selectbox(
        "Choose a scenario",
        options=list(SCENARIOS.keys()),
        index=0,
    )
    st.info(f"**Scene:** The {product_type} is {SCENARIOS[scenario_name]}")

    st.divider()

    # ── 4. Cost / Rate-limit Warning ──────────────────────────────────────────
    st.warning(
        "**Free Tier Rate Limits — please read before generating.**\n\n"
        "- Gemini 2.5 Flash: ~10 RPM / 250,000 TPM on the free tier.\n"
        "- Imagen 3: limited free quota — each generation consumes quota.\n"
        "- Generating repeatedly in quick succession may exhaust your free allowance "
        "or trigger temporary blocks.\n\n"
        "Check your usage at **Google AI Studio → API usage**."
    )

    confirmed = st.checkbox(
        "I understand the API rate limits and will not generate excessively."
    )

    generate_btn = st.button(
        "Generate Lifestyle Image",
        type="primary",
        disabled=not confirmed or not uploaded_files,
        use_container_width=True,
    )

    if not uploaded_files:
        st.caption("Upload at least one product image to enable generation.")
    elif not confirmed:
        st.caption("Check the confirmation box above to enable the button.")

# ── Right column: output ──────────────────────────────────────────────────────
with right:
    st.subheader("4  Generated Image")

    if "product_description" not in st.session_state:
        st.session_state.product_description = None
    if "last_files" not in st.session_state:
        st.session_state.last_files = []

    if generate_btn:
        # Detect if uploaded files changed → re-analyze
        current_file_names = [f.name for f in uploaded_files]
        files_changed = current_file_names != st.session_state.last_files

        if files_changed or st.session_state.product_description is None:
            with st.spinner("Step 1/2 — Analyzing product images with Gemini..."):
                try:
                    # Reset file read positions
                    for f in uploaded_files:
                        f.seek(0)
                    description = analyze_product_images(client, uploaded_files, product_type)

                    st.session_state.product_description = description
                    st.session_state.last_files = current_file_names
                except Exception as e:
                    st.error(f"Image analysis failed: {e}")
                    st.stop()
        else:
            description = st.session_state.product_description

        with st.expander("Product Description (auto-generated)", expanded=False):
            st.write(st.session_state.product_description)

        final_prompt = build_final_prompt(
            st.session_state.product_description,
            product_type,
            SCENARIOS[scenario_name],
            lighting,
            mood,
        )

        with st.expander("Final Prompt sent to Imagen 3", expanded=False):
            st.code(final_prompt, language=None)

        with st.spinner("Step 2/2 — Generating lifestyle image with Imagen 3..."):
            try:
                image_bytes = generate_lifestyle_image(client, final_prompt, negative_prompt)
            except Exception as e:
                st.error(f"Image generation failed: {e}")
                st.stop()

        st.success("Image generated successfully!")
        st.image(image_bytes, use_container_width=True)
        st.download_button(
            label="Download Image",
            data=image_bytes,
            file_name=f"lifestyle_{scenario_name.lower().replace(' ', '_')}.png",
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
                Your generated image will appear here
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.product_description:
            with st.expander("Cached Product Description (from previous run)", expanded=False):
                st.write(st.session_state.product_description)
