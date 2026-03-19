# **Resolvent model-based analyses of coherent structures in Langmuir turbulence**

**Authors:** Anqing Xuan and Lian Shen  
**Affiliation:** Department of Mechanical Engineering and Saint Anthony Falls Laboratory, University of Minnesota, Minneapolis, MN 55455, USA  
**Journal:** J. Fluid Mech. (2025), vol. 1023, A4  
**DOI:** [10.1017/jfm.2025.10830](https://doi.org/10.1017/jfm.2025.10830)

## **Abstract**

We present an analysis of the coherent structures in Langmuir turbulence, a state of the ocean surface boundary layer driven by the interactions between water waves and wind-induced shear, via a resolvent framework. Langmuir turbulence is characterised by multiscale vortical structures, notably counter-rotating roll pairs known as Langmuir circulations. While classic linear stability analyses of the Craik-Leibovich equations have revealed key instability mechanisms underlying Langmuir circulations, the present work incorporates the turbulent mean state and varying eddy viscosity using data from large-eddy simulations (LES) to investigate the turbulence dynamics of fully developed Langmuir turbulence. Scale-dependent resolvent analyses reveal a new formation mechanism of two-dimensional circulating rolls and three-dimensional turbulent coherent vortices through linear amplification of sustained harmonic forcing. Moreover, the integrated energy spectra predicted by the principal resolvent modes in response to broadband harmonic forcing capture the dominant spanwise length scales that are consistent with the LES data. These results demonstrate the feasibility of resolvent analyses in capturing key features of multiscale turbulence-wave interactions in the statistical stationary state of Langmuir turbulence.

## **1\. Introduction**

The ocean surface boundary layer features characteristic circulating rolls known as Langmuir circulations, which are formed by the combined influence of wind-driven currents and surface waves. These circulations organize into pairs of counter-rotating rolls, inducing alternating converging and diverging regions at the water surface, leading to the accumulation of buoyant materials into visible streaks known as windrows.

### **The Craik-Leibovich (CL) Model**

Early theoretical work by Craik & Leibovich (1976) developed the CL model to describe the evolution of ocean currents under the influence of surface gravity waves. Based on multiscale asymptotic analysis, it filters out oscillating wave motions and retains slowly varying currents. The wave effect is represented by a **vortex force**, defined as the cross product of the wave's Stokes drift and the flow vorticity:  
$U^s \\times (\\nabla \\times v)$.

### **Limitations of Classic Stability Analysis**

Classic linear stability analyses focus on the onset of Langmuir circulations (CL2 instability) by examining the most unstable eigenmode. However, they:

1. Often assume idealized mean velocity profiles and constant eddy viscosity.  
2. Focus primarily on two-dimensional rolls with infinite streamwise lengths.  
3. Overlook the multiscale, three-dimensional nature of fully developed Langmuir turbulence.

### **Resolvent Analysis Approach**

Resolvent analysis examines a linearized system's response to harmonic input forcing around a base flow. Unlike stability analysis, it can be applied to fully developed, statistically stationary turbulent flows. The base flow is the ensemble mean flow, and nonlinear interactions are treated as a superposition of harmonic forcing modes.

## **2\. Formulation and LES Data**

### **2.1. Linear Model for Fluctuations**

The wave-averaged momentum equations are given by:

$$\\frac{\\partial v}{\\partial t} \+ (v^L \\cdot \\nabla)v \= \-\\nabla \\Phi \+ \\nu \\Delta v \- v \\times (\\nabla \\times v^L) \- (v \\cdot \\nabla)v^L$$  
where $v$ is Eulerian velocity, $U^s$ is Stokes drift, $v^L \= v \+ U^s$ is Lagrangian velocity, and $\\Phi$ is modified pressure.  
Decomposing flow into mean and perturbation ($v \= U \+ u$) and incorporating a vertically varying eddy viscosity $\\nu\_T$:

$$\\frac{\\partial u}{\\partial t} \+ (U^L \\cdot \\nabla)u \+ (u \\cdot \\nabla)U^L \+ u \\times (\\nabla \\times U^s) \= \-\\nabla p \+ \\nu \\nabla \\cdot \\left\[ \\frac{\\nu\_T}{\\nu} (\\nabla u \+ \\nabla u^T) \\right\] \+ d$$$$\\nabla \\cdot u \= 0$$

#### **Pressure-Eliminated Formulation**

In terms of vertical perturbed velocity $v$ and vertical vorticity $\\omega\_y$, the system becomes:

$$\\frac{\\partial \\Delta v}{\\partial t} \+ U^L \\frac{\\partial \\Delta v}{\\partial x} \- U'' \\frac{\\partial v}{\\partial x} \+ U^{s'} \\frac{\\partial \\omega\_y}{\\partial z} \- \\nu\_T \\Delta^2 v \- 2\\nu\_T' \\frac{\\partial \\Delta v}{\\partial y} \- 2\\nu\_T'' \\left( \-\\frac{\\partial^2 v}{\\partial x^2} \+ \\frac{\\partial^2 v}{\\partial y^2} \- \\frac{\\partial^2 v}{\\partial z^2} \\right) \= \\mathcal{F}\_v(d)$$$$\\frac{\\partial \\omega\_y}{\\partial t} \+ U^L \\frac{\\partial \\omega\_y}{\\partial x} \+ U' \\frac{\\partial v}{\\partial z} \- \\nu\_T \\Delta \\omega\_y \- \\nu\_T' \\frac{\\partial \\omega\_y}{\\partial y} \= \\mathcal{F}\_{\\omega}(d)$$

#### **Resolvent Operator**

Transforming to Fourier space with state vector $\\hat{\\xi} \= \[\\hat{v}, \\hat{\\omega}\_y\]^T$:

$$-(i\\omega E \+ F)\\hat{\\xi} \= B\\hat{d}$$  
The transfer operator $T$ relates input forcing $\\hat{d}$ to velocity response $\\hat{u}$:

$$\\hat{u} \= T\\hat{d}, \\quad T \= C(i\\omega E \- F)^{-1}B$$  
Schmidt decomposition provides the resolvent modes:

$$T\\hat{d} \= \\sum\_{j=1}^{\\infty} \\sigma\_j \\langle \\hat{d}, \\phi\_j \\rangle\_E \\Psi\_j$$

### **2.2. LES Setup**

* **Domain:** $L\_x \\times L\_y \\times L\_z \= 8\\pi H \\times H \\times 4\\pi H$.  
* **Langmuir Number:** $La\_t \= \\sqrt{u\_\* / U\_0^s} \= 0.2$ and $0.3$.  
* **Reynolds Number:** $Re\_\\tau \= u\_\* H / \\nu \= 1000$.  
* **Eddy Viscosity:** Calculated as $\\nu\_t(y) \= \-\\overline{u'v'} / (dU^L/dy)$.

## **3\. Results**

### **3.1. Harmonic Responses and Coherent Structures**

The optimal energy amplification $G$ is given by the square of the largest singular value: $G \= \\sigma\_1^2$.

#### **Two Regimes of Amplification**

1. **Very-large-scale motions (**$\\lambda\_z \> H$**):** Maximum amplification occurs for streamwise-invariant modes ($k\_x \= 0$). These correspond to full-depth Langmuir cells.  
2. **Smaller-scale motions (**$\\lambda\_z \< H$**):** Strong amplification for a range of large streamwise wavelengths. These are near-surface three-dimensional vortices.

#### **Key Findings on Structures**

* **Langmuir Cells (**$k\_x \= 0$**):** Consist of counter-rotating rolls. Forcing is dominated by the streamwise component $d\_x$.  
* **3D Turbulent Vortices:** Quasi-streamwise vortices concentrated near the surface, inclined in the downstream direction. They are highly sensitive to nonlinear forcing.

### **3.2. Energy Distribution**

The system exhibits a **low-rank structure**, meaning a few leading resolvent modes dominate the energy content.

* **Vertical Velocity Spectrum:** The model accurately predicts the peak spanwise length scales of vertical velocity fluctuations seen in LES.  
* **Depth Dependence:** Wavelength regions where the system is strongly low-rank shift toward larger-scale motions with increasing depth, matching LES trends.

## **4\. Conclusions and Discussion**

1. **New Mechanism:** Langmuir vortical structures can be sustained by the linear amplification of harmonic forcing originating from turbulence nonlinearity, complementing the CL-II instability mechanism.  
2. **Multiscale Capture:** Resolvent analysis successfully captures both large-scale rolls and smaller-scale near-surface vortices.  
3. **Predictive Capability:** Integrated response to broadband forcing reproduces key features of the vertical velocity spectrum.  
4. **Future Work:** Suggestions include extending the system into a wave-phase-resolved framework and using data-driven methods to refine forcing/eddy viscosity models.

## **Appendix: Key Equations & Operators**

**Operator E:**

$$E \= \\begin{bmatrix} \\hat{\\Delta} & 0 \\\\ 0 & I \\end{bmatrix}$$  
**Operator F:**

$$F \= \\begin{bmatrix} \\mathcal{L}\_{OS} & \-ik\_z U^{s'} \\\\ \-ik\_z U' & \\mathcal{L}\_{Sq} \\end{bmatrix}$$  
**Boundary Conditions (Stress-free):**

$$\\hat{v} \= \\mathcal{D}^2 \\hat{v} \= \\mathcal{D} \\hat{\\omega}\_y \= 0 \\quad \\text{at } y=0, \-H$$  
**Vertical Velocity Energy Spectrum:**

$$E\_v(y; k\_z) \= \\iint k\_x^2 k\_z (\\sigma\_1 |v\_1(y)|)^2 d \\log(k\_x) dc$$