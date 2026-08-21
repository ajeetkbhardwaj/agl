# Algebraic Geometric Learning

In standard ANNs we uses the Linear Function for Input scale and transformation but OrthoPolyNNs will use the special orthogonal polynomials to see what advantages i can provide ?

Real-World Problems × Your Package's Capabilities

1. Kepler's Third Law — discovered from raw exoplanet data ⭐ flagship

- Math: Symbolic regression / physical-law discovery. Fit log(T) vs log(a) with a PNN, then get_global_equation() should print out something like -0.667 + 1.5*log_a — i.e., the network hands you back T² ∝ a³ as a closed-form equation.
- Module: PolynomialNeuralNetwork + global equation extractor (+ round_to=None exactness check)
- Data (verified working, no registration): NASA Exoplanet Archive — orbital period vs semi-major axis for ~5000 confirmed planets:
  https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_orbper,pl_orbsmax+from+pscomppars+where+pl_orbper+is+not+null+and+pl_orbsmax+is+not+null&format=csv

2. Mauna Loa CO₂ — trend + seasonal structure

- Math: Function approximation on a bounded domain with orthogonal expansions; how Chebyshev networks extrapolate vs fail beyond the training window (the clamp story from Part-I §7 becomes visible).
- Module: OrthoPolyNetwork (additive mode, adaptive normalization)
- Data (verified): https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv

3. Sunspot Cycles — sharp, spiky, unbounded autoregression

- Math: Predict next-month activity from the last k months; Hermite (unbounded interval) vs Chebyshev comparison, and whether RationalPolyLayer's localized poles capture cycle peaks better than global polynomials.
- Module: OrthoPolyNetwork (hermite basis) vs RationalPolyLayer
- Data (verified): https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt

4. Concrete Compressive Strength — interpretable engineering formula

- Math: 1030 samples, 8 ingredients → MPa. Train PNN, extract the closed-form mix-design formula, verify it matches forward numerically; use spectral regularization as ingredient-importance ranking.
- Module: PNN + extractor + OrthoPolyTrainer spectral reg
- Data (verified): https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip (page: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

5. Combined Cycle Power Plant — large-scale smooth physics

- Math: 9568 sensor readings → electrical output; additive vs multiplicative (Segre) interaction modes at scale, benchmarked against a plain PyTorch MLP.
- Module: OrthoPolyNetwork, both interaction modes
- Data: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant

6. Airfoil Self-Noise — scaling-law discovery (NASA data)

- Math: 1503 wind-tunnel measurements → sound pressure; extract a symbolic scaling law, then use GroebnerLayer/IdealMembershipLoss to hard-enforce a known aerodynamic relation during training and measure the accuracy gain.
- Module: GroebnerLayer, IdealMembershipLoss, PolynomialAwareOptimizer
- Data: https://archive.ics.uci.edu/dataset/291/airfoil+self+noise
