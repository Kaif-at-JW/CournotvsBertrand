# Game Theory for Data Science Project
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time


st.set_page_config(page_title="Pricing Competition: Cournot vs Bertrand",
                   layout="wide")
st.title("Pricing Competition: Cournot vs Bertrand")


st.sidebar.header("Model parameters")
a = st.sidebar.slider("Demand intercept a", min_value=10.0, max_value=200.0, value=60.0, step=1.0)
b = st.sidebar.slider("Demand slope b", min_value=0.05, max_value=5.0, value=1.0, step=0.05)
c1 = st.sidebar.slider("Firm 1 marginal cost c1", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
c2 = st.sidebar.slider("Firm 2 marginal cost c2", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
model = st.sidebar.selectbox("Competition model", ["Cournot (quantities)", "Bertrand (prices)"])

st.sidebar.markdown("---")
st.sidebar.header("Dynamics & visualization")
steps = st.sidebar.slider("Best-response iterations", min_value=5, max_value=200, value=40, step=5)
init_val1 = st.sidebar.number_input("Initial strategy firm 1 (q or p)", value=10.0, step=1.0)
init_val2 = st.sidebar.number_input("Initial strategy firm 2 (q or p)", value=10.0, step=1.0)
run_dynamics = st.sidebar.button("Run best-response dynamics")
show_hints = st.sidebar.checkbox("Show hints / explanations", value=True)
graph_choice = st.sidebar.multiselect("Graphs to show", 
                                      ["Profits", "Reaction functions", "Best-response dynamics", "Demand / Price-Quantity"],
                                      default=["Profits", "Reaction functions", "Best-response dynamics"])

st.markdown("## Model summary")
if show_hints:
    st.markdown(
        """
        **Cournot model**: Firms simultaneously choose quantities q1 and q2. Market price is P = a - b (q1+q2).
        Profit of firm i: π_i = (P - c_i) q_i.

        **Bertrand model (simplified)**: Firms choose prices p1 and p2. Consumers buy from the cheaper firm.
        If prices are equal, demand is split. Demand function (if firm wins) is D(p) = max(a - b p, 0).
        The app uses a continuous best-response rule: maximize own profit assuming you capture demand when you undercut,
        and split demand when equal.
        """
    )

def cournot_equilibrium(a, b, c1, c2):
    # q_i^* = (a - 2 c_i + c_j) / (3 b)
    q1 = (a - 2*c1 + c2) / (3*b)
    q2 = (a - 2*c2 + c1) / (3*b)
    q1 = max(q1, 0.0)
    q2 = max(q2, 0.0)
    P = max(a - b*(q1 + q2), 0.0)
    pi1 = (P - c1) * q1
    pi2 = (P - c2) * q2
    return P, q1, q2, pi1, pi2

def cournot_best_response(a, b, ci, qj):
    qi = (a - ci - b * qj) / (2 * b)
    return max(qi, 0.0)

def bertrand_best_response(a, b, ci, pj):
    """
    Given rival price pj, compute best response price in [ci, pj] (if you undercut, you capture demand).
    Profit when p < pj: (p - ci) (a - b p)
    The unconstrained optimum of f(p) = (p-ci)(a - b p) is p* = (a + ci) / (2 b).
    Choose p_best = clamp(p*, ci, pj - eps) if p* < pj, else choose slightly undercut pj or pj if equal costs.
    """
    eps = 1e-3
    p_star = (a + ci) / (2 * b)
    if p_star < ci:
        p_star = ci
    if p_star + 1e-9 < pj:
        return p_star
    candidate = max(ci, pj - eps)
    if candidate >= pj:
        return pj
    return candidate

def demand_at_price(a, b, p):
    return max(a - b * p, 0.0)

def profit_cournot(a, b, c, q1, q2, i):
    P = max(a - b * (q1 + q2), 0.0)
    if i == 1:
        return (P - c) * q1
    else:
        return (P - c) * q2

def profit_bertrand(a, b, p, c, rival_p, firm_index):
    if p < rival_p - 1e-9:
        D = demand_at_price(a, b, p)
        return (p - c) * D
    elif p > rival_p + 1e-9:
        return 0.0
    else:
        D = demand_at_price(a, b, p) / 2.0
        return (p - c) * D


if model.startswith("Cournot"):
    P_eq, q1_eq, q2_eq, pi1_eq, pi2_eq = cournot_equilibrium(a, b, c1, c2)
    st.markdown("### Static/Cournot results")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.metric("Equilibrium price P", f"{P_eq:.3f}")
    with col2:
        st.metric("Firm 1 quantity q1*", f"{q1_eq:.3f}")
    with col3:
        st.metric("Firm 2 quantity q2*", f"{q2_eq:.3f}")
    col4, col5 = st.columns([1,1])
    with col4:
        st.metric("Firm 1 profit π1*", f"{pi1_eq:.3f}")
    with col5:
        st.metric("Firm 2 profit π2*", f"{pi2_eq:.3f}")

else:
   
    if abs(c1 - c2) < 1e-9:
        p1_eq = p2_eq = c1
        pi1_eq = pi2_eq = 0.0
    else:
        if c1 < c2:
            p_star = (a + c1) / (2 * b)
            p1_eq = p_star if p_star < c2 else (c2 - 1e-3)
            p2_eq = c2
            pi1_eq = profit_bertrand(a, b, p1_eq, c1, p2_eq, 1)
            pi2_eq = 0.0
        else:
            p_star = (a + c2) / (2 * b)
            p2_eq = p_star if p_star < c1 else (c1 - 1e-3)
            p1_eq = c1
            pi2_eq = profit_bertrand(a, b, p2_eq, c2, p1_eq, 2)
            pi1_eq = 0.0

    st.markdown("### Static/Bertrand results")
    col1, col2 = st.columns([1,1])
    with col1:
        st.metric("Firm 1 price p1*", f"{p1_eq:.3f}")
    with col2:
        st.metric("Firm 2 price p2*", f"{p2_eq:.3f}")
    c3, c4 = st.columns([1,1])
    with c3:
        st.metric("Firm 1 profit π1*", f"{pi1_eq:.3f}")
    with c4:
        st.metric("Firm 2 profit π2*", f"{pi2_eq:.3f}")

st.markdown("---")


left_col, right_col = st.columns([2, 1])

with left_col:
    if "Profits" in graph_choice:
        fig1, ax1 = plt.subplots(figsize=(6,3))
        if model.startswith("Cournot"):
            ax1.bar(["Firm 1", "Firm 2"], [pi1_eq, pi2_eq])
            ax1.set_ylabel("Profit")
            ax1.set_title("Equilibrium profits (Cournot)")
        else:
            ax1.bar(["Firm 1", "Firm 2"], [pi1_eq, pi2_eq])
            ax1.set_ylabel("Profit")
            ax1.set_title("Equilibrium profits (Bertrand)")
        st.pyplot(fig1)
        plt.close(fig1)

    if "Reaction functions" in graph_choice:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        if model.startswith("Cournot"):
            q_grid = np.linspace(0, max( (a - min(c1,c2)) / b, 1.0), 300)
            br1 = [cournot_best_response(a, b, c1, qj) for qj in q_grid]
            br2 = [cournot_best_response(a, b, c2, qj) for qj in q_grid]
            ax2.plot(q_grid, br1, label="BR1(q2)")
            ax2.plot(q_grid, br2, label="BR2(q1)")
            ax2.set_xlabel("Opposing quantity")
            ax2.set_ylabel("Best response quantity")
            # Equilibrium point
            ax2.scatter([q2_eq], [q1_eq], color='black', zorder=5, label="Equilibrium")
            ax2.legend()
            ax2.set_title("Reaction functions and Cournot equilibrium")
        else:
            pmax = max(a / b, max(c1, c2) + 10)
            p_grid = np.linspace(0.0, pmax, 400)
            br1 = [bertrand_best_response(a, b, c1, pj) for pj in p_grid]
            br2 = [bertrand_best_response(a, b, c2, pj) for pj in p_grid]
            ax2.plot(p_grid, br1, label="BR1(p2)")
            ax2.plot(p_grid, br2, label="BR2(p1)")
            ax2.scatter([p2_eq], [p1_eq], color='black', zorder=5, label="Equilibrium")
            ax2.set_xlabel("Opposing price")
            ax2.set_ylabel("Best response price")
            ax2.legend()
            ax2.set_title("Reaction functions and Bertrand equilibrium (approx.)")
        st.pyplot(fig2)
        plt.close(fig2)

    if "Demand / Price-Quantity" in graph_choice:
        fig3, ax3 = plt.subplots(figsize=(6,3))
        if model.startswith("Cournot"):
            total_q = np.linspace(0, max(q1_eq+q2_eq, a/b), 200)
            P_of_Q = np.maximum(a - b * total_q, 0)
            ax3.plot(total_q, P_of_Q, label="Inverse demand P(Q)")
            ax3.scatter([q1_eq + q2_eq], [P_eq], color='red', label="Equilibrium total Q,P")
            ax3.set_xlabel("Total quantity Q")
            ax3.set_ylabel("Price P")
            ax3.set_title("Demand curve and Cournot equilibrium")
            ax3.legend()
        else:
            p_grid = np.linspace(0, max(a/b, p1_eq, p2_eq), 200)
            D = [demand_at_price(a, b, p) for p in p_grid]
            ax3.plot(p_grid, D, label="Demand D(p)")
            ax3.scatter([p1_eq, p2_eq], [demand_at_price(a, b, p1_eq), demand_at_price(a, b, p2_eq)],
                        color=['tab:orange','tab:green'], label="Equilibrium prices")
            ax3.set_xlabel("Price p")
            ax3.set_ylabel("Demand D(p)")
            ax3.set_title("Demand and Bertrand equilibrium prices")
            ax3.legend()
        st.pyplot(fig3)
        plt.close(fig3)


with right_col:
    st.subheader("Best-response dynamics")
    st.write("Use the 'Run best-response dynamics' button in the sidebar to animate stepwise convergence.")
    st.write("Initial strategies are taken from the inputs 'Initial strategy firm 1/2'.")

    if run_dynamics:
        placeholder = st.empty()
        x1 = float(init_val1)
        x2 = float(init_val2)
        history1 = [x1]
        history2 = [x2]

        for k in range(steps):
            if model.startswith("Cournot"):
                x1 = cournot_best_response(a, b, c1, x2)
                x2 = cournot_best_response(a, b, c2, x1)
                history1.append(x1)
                history2.append(x2)

                fig, ax = plt.subplots(figsize=(5,4))
                ax.plot(history1, label="q1 (firm1)")
                ax.plot(history2, label="q2 (firm2)")
                ax.hlines(q1_eq, 0, steps, linestyles='--', linewidth=0.8, label="q1*")
                ax.hlines(q2_eq, 0, steps, linestyles='-.', linewidth=0.8, label="q2*")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Quantity")
                ax.set_title("Best-response dynamics (Cournot)")
                ax.legend()
                placeholder.pyplot(fig)
                plt.close(fig)

            else:
                p1 = bertrand_best_response(a, b, c1, x2)
                p2 = bertrand_best_response(a, b, c2, p1)
                x1 = p1
                x2 = p2
                history1.append(x1)
                history2.append(x2)

                fig, ax = plt.subplots(figsize=(5,4))
                ax.plot(history1, label="p1 (firm1)")
                ax.plot(history2, label="p2 (firm2)")
                if 'p1_eq' in locals() and 'p2_eq' in locals():
                    ax.hlines(p1_eq, 0, steps, linestyles='--', linewidth=0.8, label="p1*")
                    ax.hlines(p2_eq, 0, steps, linestyles='-.', linewidth=0.8, label="p2*")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Price")
                ax.set_title("Best-response dynamics (Bertrand)")
                ax.legend()
                placeholder.pyplot(fig)
                plt.close(fig)

            time.sleep(0.08)  # small pause to create an animation effect

        st.markdown("**Final iterates**")
        st.write(f"Firm 1 final strategy: {history1[-1]:.4f}")
        st.write(f"Firm 2 final strategy: {history2[-1]:.4f}")
        if model.startswith("Cournot"):
            st.write(f"Static Cournot equilibrium q1* = {q1_eq:.4f}, q2* = {q2_eq:.4f}")
        else:
            st.write(f"Static Bertrand (approx.) p1* = {p1_eq:.4f}, p2* = {p2_eq:.4f}")

    else:
        st.info("Press 'Run best-response dynamics' in the sidebar to animate.")

st.markdown("---")
st.caption("Made by Kaif Satopay - Game Theory for Data Science Mini Project")
