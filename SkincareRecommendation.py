import streamlit as st
import traceback

from SearchQuery import (
    recommend_top_20_products,
    make_review_insights,
)

# =========================================================
# Helpers
# =========================================================
def snip(text: str, n: int = 220) -> str:
    if not text:
        return ""
    t = text.replace("\n", " ").strip()
    return t[:n] + ("..." if len(t) > n else "")


# =========================================================
# UI helpers
# =========================================================
def render_product_card(p: dict, title: str = "") -> None:
    if title:
        st.markdown(f"### {title}")

    left, right = st.columns([1, 2])

    with left:
        img = p.get("image_url")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.caption("No image available")

    with right:
        st.markdown(f"#### {p.get('brand','')} — {p.get('name','')}")
        st.write(f"**Category:** {p.get('category','')}")
        st.write(f"**Subcategory:** {p.get('subcategory','')}")
        st.write(f"**Price:** {p.get('price','')}")
        st.write(f"**Size:** {p.get('size','')}")
        st.write(f"**Skin type:** {p.get('skin_type','')}")
        st.write(f"**Skin concerns:** {p.get('skin_concerns','')}")

    # ---- expandable product info ----
    with st.expander("Show more"):
        desc = p.get("description") or ""
        ing = p.get("important_ingredients") or ""
        how = p.get("how_to_use") or ""

        if desc:
            st.markdown("**Description**")
            st.write(desc)
        else:
            st.caption("No description.")

        st.markdown("---")

        if ing:
            st.markdown("**Important ingredients**")
            st.write(ing)
        else:
            st.caption("No ingredients listed.")

        st.markdown("---")

        if how:
            st.markdown("**How to use**")
            st.write(how)
        else:
            st.caption("No how-to-use instructions.")

    # =====================================================
    # Review insights (ON DEMAND)
    # =====================================================
    pid = p.get("product_id") or ""

    if st.button(
        "Generate review insights",
        key=f"insights_{pid}_{title}",
    ):
        if pid not in st.session_state.insights_cache:
            with st.spinner("Analyzing customer feedback…"):
                st.session_state.insights_cache[pid] = make_review_insights(
                    p,
                    p.get("all_reviews") or [],
                )

    if pid in st.session_state.insights_cache:
        pack = st.session_state.insights_cache[pid]

        st.markdown("**Review summary**")
        st.write(pack.get("review_summary", ""))

        st.markdown("**Pros**")
        st.write(pack.get("pros_summary", ""))

        st.markdown("**Cons**")
        st.write(pack.get("cons_summary", ""))

    st.divider()


# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Skincare Recommender", layout="wide")
st.title("Skincare Product Recommender")

# =========================================================
# Session state init
# =========================================================
if "last_out" not in st.session_state:
    st.session_state.last_out = None

if "last_query" not in st.session_state:
    st.session_state.last_query = "cleanser from la roche posay under $30"

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "insights_cache" not in st.session_state:
    st.session_state.insights_cache = {}  # product_id -> insights


# =========================================================
# Sidebar (INPUT ONLY)
# =========================================================
with st.sidebar:
    st.header("Search")

    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "Query",
            value=st.session_state.last_query,
            help="Example: cleanser from la roche posay under $30",
        )
        show_top3 = st.checkbox("Show Top 3", value=False)
        submitted = st.form_submit_button("Recommend")

    if submitted:
        st.session_state.last_query = query
        st.session_state.pending_query = query


# =========================================================
# Backend call
# =========================================================
if st.session_state.pending_query:
    q = st.session_state.pending_query
    st.session_state.pending_query = None

    try:
        with st.spinner("Finding the best products for you…"):
            st.session_state.last_out = recommend_top_20_products(q)
    except Exception:
        st.error("Backend crashed:")
        st.code(traceback.format_exc())
        st.stop()


out = st.session_state.last_out

if out is None:
    st.info("Enter a query and click **Recommend**.")
    st.stop()

if out.get("message"):
    st.warning(out["message"])
    st.stop()

best = out.get("best_product")
stage_b = out.get("stage_b") or []

# =========================================================
# Best product
# =========================================================
st.subheader("Recommended product")

if not best:
    st.error("No best product returned.")
    st.stop()

render_product_card(best, title="Best match")

# =========================================================
# Top 3 (optional)
# =========================================================
if show_top3:
    st.subheader("Top 3 candidates")

    if not stage_b:
        st.info("Top 3 list is empty.")
    else:
        for idx, p in enumerate(stage_b[:3], 1):
            render_product_card(p, title=f"#{idx}")
