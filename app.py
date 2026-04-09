from pathlib import Path
import logging
import time

import clip
import torch
import joblib
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="Multimodal Misinformation Detector")

# Defines robust paths so this app works on Streamlit Cloud.
BASE_DIR = Path(__file__).resolve().parent          # .../demo
PROJECT_ROOT = BASE_DIR.parent                      # .../(project root)
ADAPTED_MODEL_PATH = BASE_DIR / "adapted_model.pkl"
BASELINE_MODEL_PATH = BASE_DIR / "model.pkl"


def resolve_model_path() -> Path:
    """Prefer the adapted model when it has been exported, otherwise fall back."""
    if ADAPTED_MODEL_PATH.exists():
        return ADAPTED_MODEL_PATH
    return BASELINE_MODEL_PATH


MODEL_PATH = resolve_model_path()

DEVICE = "cpu"
MISINFO_LABEL = 0

logger = logging.getLogger("mbd_app")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def confidence_band(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    if prob < 0.66:
        return "Medium"
    return "High"


def risk_message(prob: float) -> str:
    if prob < 0.33:
        return "Low risk. The image and text look mostly consistent."
    if prob < 0.66:
        return "Medium risk. Some signs are mixed, so this should be reviewed."
    return "High risk. The model sees strong misinformation patterns."


def confidence_message(prob: float) -> str:
    conf = abs(prob - 0.5) * 2
    if conf < 0.33:
        return "Low confidence"
    if conf < 0.66:
        return "Moderate confidence"
    return "High confidence"


@st.cache_resource(show_spinner=False)
def load_clip_model(device: str):
    """Load CLIP model with proper error handling"""
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        logger.info("CLIP loaded successfully")
        return model, preprocess
    except Exception as e:
        logger.error(f"Failed to load CLIP: {e}")
        st.error(f"Failed to load CLIP model: {e}")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_classifier(model_path: str):
    """Load trained classifier"""
    try:
        model = joblib.load(model_path)
        logger.info("Classifier loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        st.error(f"Failed to load classifier: {e}")
        st.stop()


def prepare_features(image: Image.Image, text: str, clip_model, preprocess):
    """Prepare features exactly like training"""
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])

    with torch.no_grad():
        img_emb = clip_model.encode_image(image_input)
        txt_emb = clip_model.encode_text(text_input)

    # Normalizes embeddings before feature construction.
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    # Matches training features: cosine similarity, absolute difference, and concatenation.
    # Produces 1537 features in total for ViT-B/32 embeddings.
    cos_sim = torch.nn.functional.cosine_similarity(img_emb, txt_emb, dim=1).unsqueeze(1)
    abs_diff = torch.abs(img_emb - txt_emb)
    concat = torch.cat([img_emb, txt_emb], dim=1)
    features = torch.cat([cos_sim, abs_diff, concat], dim=1).cpu().numpy()
    return features


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def get_misinfo_class_index(classifier) -> int:
    if not hasattr(classifier, "classes_"):
        raise ValueError("Classifier exposes predict_proba but not classes_.")

    classes = list(classifier.classes_)
    if MISINFO_LABEL not in classes:
        raise ValueError(
            f"Misinformation label {MISINFO_LABEL} missing from classifier classes: {classes}"
        )

    return classes.index(MISINFO_LABEL)


def predict_misinfo_probability(features, classifier) -> float:
    if not hasattr(classifier, "predict_proba"):
        raise ValueError("Classifier does not support predict_proba.")

    misinfo_idx = get_misinfo_class_index(classifier)
    proba = classifier.predict_proba(features)[0]
    return float(proba[misinfo_idx])


def linear_explain(features_np: np.ndarray, classifier):
    """
    Exact linear explanation for linear classifiers like LogisticRegression.
    Returns probability/logit for the classifier's positive class (classes_[1] for binary).
    Returns: prob, logit, bias, per_feature_contrib, group_contribs(dict)
    """
    if not (hasattr(classifier, "coef_") and hasattr(classifier, "intercept_")):
        raise ValueError("Classifier does not expose linear coefficients.")

    n_features = int(features_np.shape[1])
    if n_features <= 1 or (n_features - 1) % 3 != 0:
        raise ValueError(f"Unexpected feature layout: got {n_features}, expected 1 + 3*d.")

    emb_dim = (n_features - 1) // 3
    w = classifier.coef_.reshape(-1)
    b = float(classifier.intercept_.reshape(-1)[0])
    x = features_np.reshape(-1)

    if w.shape[0] != x.shape[0]:
        raise ValueError(
            f"Coefficient/feature mismatch: coef={w.shape[0]} features={x.shape[0]}"
        )

    contrib = w * x
    logit = b + float(contrib.sum())
    prob = sigmoid(logit)

    cos_idx = slice(0, 1)
    abs_idx = slice(1, 1 + emb_dim)
    img_idx = slice(1 + emb_dim, 1 + emb_dim + emb_dim)
    txt_idx = slice(1 + emb_dim + emb_dim, 1 + emb_dim + 2 * emb_dim)

    group_contribs = {
        "bias": b,
        "cos_sim": float(contrib[cos_idx].sum()),
        "abs_diff": float(contrib[abs_idx].sum()),
        "img_emb": float(contrib[img_idx].sum()),
        "txt_emb": float(contrib[txt_idx].sum()),
        "total_logit": logit,
    }
    return prob, logit, b, contrib, group_contribs


def get_positive_class_label(classifier):
    """Return the class associated with the linear logit/sigmoid direction."""
    classes = getattr(classifier, "classes_", None)
    if classes is None or len(classes) < 2:
        return None
    return int(list(classes)[-1])


def top_k_contribs(contrib: np.ndarray, k: int = 8):
    """Return top positive and negative per-dimension contributions."""
    if contrib.size == 0:
        return np.array([], dtype=int), np.array([]), np.array([], dtype=int), np.array([])

    k = max(1, min(k, contrib.size))
    idx_sorted = np.argsort(contrib)
    neg = idx_sorted[:k]
    pos = idx_sorted[-k:][::-1]
    return pos, contrib[pos], neg, contrib[neg]


def word_influence_loo(
    image: Image.Image,
    text: str,
    clip_model,
    preprocess,
    classifier,
    max_words: int = 25,
):
    """
    Approximate token influence with leave-one-word-out.
    Positive delta means the removed word increased misinformation probability.
    """
    if not hasattr(classifier, "predict_proba"):
        return []

    words = text.strip().split()
    if not words:
        return []

    words = words[:max_words]
    base_text = " ".join(words)
    base_feat = prepare_features(image, base_text, clip_model, preprocess)
    base_prob = predict_misinfo_probability(base_feat, classifier)

    impacts = []
    for i, word in enumerate(words):
        new_words = words[:i] + words[i + 1 :]
        if not new_words:
            continue
        feat_i = prepare_features(image, " ".join(new_words), clip_model, preprocess)
        prob_i = predict_misinfo_probability(feat_i, classifier)
        delta = base_prob - prob_i
        impacts.append((word, delta))

    impacts.sort(key=lambda t: abs(t[1]), reverse=True)
    return impacts[:10]


def predict_label(features, classifier):
    """Get prediction and probability"""
    pred = int(classifier.predict(features)[0])

    prob = None
    if hasattr(classifier, "predict_proba"):
        prob = predict_misinfo_probability(features, classifier)

    return pred, prob


# Builds the user interface.
st.title("Multimodal Misinformation Detector")
st.caption("Upload an image and related text to get a simple misinformation risk assessment.")
show_debug = st.sidebar.checkbox("Show debug panel", value=False)

if show_debug:
    with st.sidebar.expander("Debug", expanded=True):
        st.write(f"Device: `{DEVICE}`")
        st.write(f"Project root: `{PROJECT_ROOT}`")
        st.write(f"Model path: `{MODEL_PATH}`")
        st.write(f"Model exists: `{MODEL_PATH.exists()}`")

if not MODEL_PATH.exists():
    logger.error("Missing model file at %s", MODEL_PATH)
    st.error(f"Missing model file: {MODEL_PATH}")
    st.stop()

try:
    with st.spinner("Loading CLIP and classifier..."):
        clip_model, preprocess = load_clip_model(DEVICE)
        clf = load_classifier(str(MODEL_PATH))
        if hasattr(clf, "predict_proba"):
            _ = get_misinfo_class_index(clf)
    logger.info("Models loaded successfully. device=%s model_path=%s", DEVICE, MODEL_PATH)
except Exception as exc:
    logger.exception("Model loading failed")
    st.error(f"Failed to load models: {exc}")
    st.stop()

if show_debug:
    with st.sidebar.expander("Classifier", expanded=False):
        st.write(f"classes_: `{getattr(clf, 'classes_', 'N/A')}`")
        if hasattr(clf, "predict_proba"):
            st.write(f"misinfo class label: `{MISINFO_LABEL}`")
            st.write(f"misinfo class index: `{get_misinfo_class_index(clf)}`")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
text_input = st.text_area("Enter accompanying text", placeholder="Write the caption/post text here...")
run_clicked = st.button("Run prediction", type="primary")

if uploaded_image is not None:
    preview_col, _ = st.columns([1, 2])
    with preview_col:
        st.image(uploaded_image, caption="Uploaded image preview", width=260)

if run_clicked:
    cleaned_text = text_input.strip()

    if uploaded_image is None:
        st.warning("Please upload an image.")
        st.stop()

    if not cleaned_text:
        st.warning("Please enter non-empty text.")
        st.stop()

    try:
        image = Image.open(uploaded_image).convert("RGB")
    except UnidentifiedImageError:
        logger.warning("Rejected invalid image upload")
        st.error("The uploaded file is not a valid image.")
        st.stop()
    except Exception as exc:
        logger.exception("Image read failure")
        st.error(f"Could not read image: {exc}")
        st.stop()

    try:
        start_time = time.perf_counter()
        features = prepare_features(image, cleaned_text, clip_model, preprocess)
        pred, prob = predict_label(features, clf)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Inference complete. pred=%s prob=%s text_chars=%s elapsed_ms=%.2f",
            pred,
            "N/A" if prob is None else f"{prob:.4f}",
            len(cleaned_text),
            elapsed_ms,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        st.error(f"Inference failed: {exc}")
        st.stop()

    st.subheader("Result")

    # Shows prediction text with simple visual emphasis.
    if pred == MISINFO_LABEL:
        st.markdown("### **Potential Misinformation**")
        st.caption("This means the model found patterns often seen in misleading posts.")
    else:
        st.markdown("### **Likely Consistent**")
        st.caption("This means the image and text seem to fit together.")

    st.subheader("Risk level")
    if prob is None:
        st.info("Probability is unavailable for this classifier.")
    else:
        bounded_prob = max(0.0, min(1.0, prob))
        band = confidence_band(bounded_prob)

        # Shows a risk bar based on probability level.
        if bounded_prob > 0.66:
            st.progress(bounded_prob, text="High risk")
        elif bounded_prob > 0.33:
            st.progress(bounded_prob, text="Medium risk")
        else:
            st.progress(bounded_prob, text="Low risk")

        st.caption(f"Estimated risk score: {bounded_prob * 100:.1f}%")
        st.caption(f"Risk band: {band}")
        st.info(risk_message(bounded_prob))

    if show_debug:
        with st.sidebar.expander("Last run", expanded=True):
            st.write(f"Uploaded file: `{uploaded_image.name}`")
            st.write(f"Text length: `{len(cleaned_text)}` chars")
            st.write(f"Feature shape: `{features.shape}`")
            st.write(f"Prediction: `{pred}`")
            st.write(f"Probability: `{'N/A' if prob is None else f'{prob:.4f}'}`")
            st.write(f"Inference time: `{elapsed_ms:.2f} ms`")

    st.subheader("Why this result?")
    st.write(
        "The model checks whether the image and text support each other. "
        "If they do not match well, risk goes up."
    )

    cos_sim_value = float(features[0, 0])
    if cos_sim_value >= 0.28:
        st.caption("Image and text appear related.")
    else:
        st.caption("Image and text appear mismatched, which is a common warning sign.")

    if prob is not None:
        st.caption(f"Model certainty: {confidence_message(prob)}")

    st.subheader("What influenced this result?")
    try:
        prob_lin, logit, _, contrib, groups = linear_explain(features, clf)
        pos_class_label = get_positive_class_label(clf)
        if pos_class_label is None:
            misinfo_prob_lin = prob_lin
            pos_push_label = "misinformation"
            neg_push_label = "consistent"
        else:
            misinfo_prob_lin = prob_lin if pos_class_label == MISINFO_LABEL else (1.0 - prob_lin)
            if pos_class_label == MISINFO_LABEL:
                pos_push_label = "misinformation"
                neg_push_label = "consistent"
            else:
                pos_push_label = "consistent"
                neg_push_label = "misinformation"

        # Simple, human-readable explanation first.
        relation_effect = groups["cos_sim"]
        relation_direction = pos_push_label if relation_effect >= 0 else neg_push_label
        if relation_direction == "misinformation":
            st.write("- Image-text relation pushed the result toward misinformation risk.")
        else:
            st.write("- Image-text relation pushed the result toward consistency.")

        if abs(groups["abs_diff"]) > abs(groups["cos_sim"]):
            st.write("- Detailed embedding differences had a strong effect on this decision.")
        else:
            st.write("- Overall image-text alignment had the strongest effect on this decision.")

        st.caption(f"Approximate model score for misinformation: {misinfo_prob_lin:.2f}")

        pos_idx, pos_vals, neg_idx, neg_vals = top_k_contribs(contrib, k=8)
        with st.expander("Advanced details (technical)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Logit (raw score)", f"{logit:.3f}")
            with c2:
                st.metric("Misinformation prob (from logit)", f"{misinfo_prob_lin:.3f}")

            st.markdown("**Contribution to logit by feature group**")
            st.write(
                {
                    "bias": round(groups["bias"], 4),
                    "cos_sim": round(groups["cos_sim"], 4),
                    "abs_diff": round(groups["abs_diff"], 4),
                    "img_emb": round(groups["img_emb"], 4),
                    "txt_emb": round(groups["txt_emb"], 4),
                }
            )

            st.markdown(f"**Top + dimensions (push toward {pos_push_label})**")
            st.write(
                [
                    {"feature_idx": int(i), "logit_contrib": float(v)}
                    for i, v in zip(pos_idx, pos_vals)
                ]
            )

            st.markdown(f"**Top - dimensions (push toward {neg_push_label})**")
            st.write(
                [
                    {"feature_idx": int(i), "logit_contrib": float(v)}
                    for i, v in zip(neg_idx, neg_vals)
                ]
            )
    except Exception as exc:
        logger.info("Explainability disabled: %s", exc)
        st.info("Detailed explanation is unavailable for this model setup.")

    with st.expander("Important words in the text", expanded=False):
        try:
            impacts = word_influence_loo(
                image, cleaned_text, clip_model, preprocess, clf, max_words=25
            )
            if not impacts:
                st.write("Not enough text to estimate word influence.")
            else:
                st.caption(
                    "These are word-level effects only. The final prediction also depends on image features and overall image-text alignment."
                )
                toward_misinfo = []
                toward_consistency = []
                for w, d in impacts:
                    item = {
                        "word": w,
                        "influence": round(abs(float(d)), 4),
                        "direction": (
                            "toward misinformation"
                            if float(d) > 0
                            else "toward consistency"
                        ),
                    }
                    if float(d) > 0:
                        toward_misinfo.append(item)
                    else:
                        toward_consistency.append(item)

                if toward_misinfo:
                    st.markdown("**Words pushing toward misinformation**")
                    st.write(toward_misinfo)
                if toward_consistency:
                    st.markdown("**Words pushing toward consistency**")
                    st.write(toward_consistency)
                if toward_misinfo and toward_consistency:
                    st.caption(
                        "Mixed directions are normal: some words can support consistency while other words or the image-text mismatch still drive the final result toward misinformation."
                    )
        except Exception as exc:
            logger.info("Word influence failed: %s", exc)
            st.write("Could not compute word influence for this input.")
