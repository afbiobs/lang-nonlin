# Nonlinear Steady States to Langmuir Circulation in Shallow Layers: An Asymptotic Study

**Authors:** D. T. Hayes (a) and W. R. C. Phillips (a,b)

**(a)** Department of Mathematics, School of Science, Swinburne University of Technology, Hawthorn, Australia
**(b)** Department of Theoretical and Applied Mechanics, University of Illinois at Urbana Champaign, Urbana, IL, USA

**Journal:** Geophysical and Astrophysical Fluid Dynamics, 2017, Vol. 111, No. 1, 65–90
**DOI:** 10.1080/03091929.2016.1263302
**Received:** 22 April 2016; **Accepted:** 17 November 2016

**Keywords:** Instability theory; Shallow water waves; Wave-mean flow interactions; Boundary conditions; Langmuir circulation

---

## Abstract

The nonlinear steady states of perturbation equations describing the instability of wavy shear flows to counter-rotating vortical structures aligned with the flow in shallow water layers is considered. The structures are described by the Craik–Leibovich equations and are known as Langmuir circulation; they arise through an instability that requires the presence of shear $U'$ and differential drift $D'$ of the same sign provided a threshold Rayleigh number is exceeded. Of specific interest here is the aspect ratio of the Langmuir circulation and how that ratio is affected by nonlinearities when the layer is shallow, as in coastal waters and estuaries. For context it is known from observation that the aspect ratio (width of two cells to depth) is two to three in deep water, whereas in shallow waters it can range up to ten. Accordingly, while the ratio for deep water is well predicted by linear theory, the ratio for shallow water is not, which explains why nonlinearities are of interest. Present always, but of key importance in effecting a preferred spacing with nonzero wavenumber $l$ in shallow water Langmuir circulation, is an extra stress induced by the perturbed motion at the free surface. This stress is reflected in the boundary conditions. Moreover, since it is most influential in the limit $l \to 0$, it is instructive to expose that influence through a small-$l$ asymptotic approximation. It is found that nonlinearities ensure supercritical stability and that the critical wavenumber at onset to instability is significantly less than its linear counterpart. This results from a subcritical bifurcation due to symmetry breaking of the governing equations. The precise level of reduction is affected by the distributions of $U'$ and $D'$, but herein it ranges from 50 to 70%. This means that nonlinearities act to effect aspect ratios up to twice those given by linear theory and which are in good agreement with observation in shallow coastal waters. Finally, the extra stress at the free surface is found to play no role in the ratio of linear to nonlinear spacing but, as in the linear case, acts to ensure the spanwise wavenumber at onset to instability is nonzero.

---

## 1. Introduction

Langmuir circulation (LC) are wind aligned cross-wind periodic rolls that form beneath surface waves in open bodies of water and grow in cross-section to the size of sports stadiums. Their primary role is the formation and maintenance of the mixed layer in the upper ocean (Langmuir 1938, Craik and Leibovich 1976, Babanin et al. 2009). Although LC can form in the absence of wind (Phillips 2002), they usually form tens of minutes after the onset of winds above 3 m/s and live in the surface boundary layer over time scales long with respect to the period of the waves. LC themselves are not visible, but the convergence zones between adjacent rolls act to congregate gas bubbles and surface debris that is visible. Such markers highlight a hierarchy of cross-wind scales, of which the largest for a cell pair range from millimeters (Kenney 1993) to hundreds of meters (Plueddemann et al. 1996), possibly kilometers (Thorpe 2004), with a longitudinal (windward) extent up to fifty times their spacing (Marmorino et al. 2005).

Field measurements further indicate that the aspect ratio for a cell pair to depth is typically two to three (Smith et al. 1987, Smith 1992) both in surface layers bounded by a thermocline and when the LC extend to the bottom. But documented exceptions occur in the latter case where both visual and infrared data (Marmorino et al. 2005) indicate spacings (two cells) as large as ten times the water depth and we should like to know why. Phillips and Dai (2014) recently explored this question using linear theory and found that circumstances arise in which large aspect ratio LCs are realizable in shallow layers, albeit ratios well in excess of their observed counterparts. The object of the present work is to question the role of nonlinearities on aspect ratio in layers that are shallow, or more precisely shallow in the sense of shallow water waves, in which the lower extent of the layer, be it a thermocline or rigid, is felt by the waves.

LC result from an instability excited by waves interacting with a sheared mean flow, in a manner that Craik and Leibovich (1976) capture in a set of mean field equations known as the CL-equations. In deriving them they exploit the fact that the mean flow and LC evolve over time scales much longer than the wave period and so average over the waves, or more precisely take a Lagrangian average over the wave period (or wavelength of the waves). The rectified effect of the waves is then exposed as a mean Lagrangian velocity known as the drift (Stokes 1847) and the interaction of the drift with the evolving flow appears in the CL-equations (1) as a force term.

More recent studies derive the CL-equations (Leibovich 1980, Craik 1985, Holm 1996, Buhler 2009) from generalized Lagrangian mean theory (Andrews and McIntyre 1978), while Vladimirov et al. (2015) derive them via the method of multiple scales. Vladimirov et al. (2015) further show that the CL-equations are one of two distinguished limits for averaged oscillatory flows, a limit they term "weak vortex dynamics", in which the averaged vorticity is "frozen" into the averaged velocity plus drift. The CL-equations have restrictions, however, in that they are valid only for irrotational waves of mean slope $\epsilon$ in the presence of $O(\epsilon^2)$ or weak levels of shear. In contrast their generalized counterparts, the CLg-equations (Phillips 1998a, Phillips 2001b, Phillips and Dai 2014), are valid for rotational wave fields of any amplitude in all levels of shear.

The CL-equations expose two instability mechanisms to LC, denoted Craik–Leibovich types 1 and 2. Both require mean shear and drift. But whereas type 1 requires a drift field that varies cross-wind, type 2 does not. In Nature, of course, there is no limitation on the combinations of surface waves that arise, but common is a random spectrum of waves and therein cross-wind variations in drift phase average out (Craik and Leibovich 1976, Phillips 2001b) leaving a drift field able to excite the type 2 or CL2 instability, which is an (initially) exponentially growing wave-driven inviscid centrifugal instability (Phillips 1998a). CL2 has received wide attention, albeit largely in the context of deep water waves (Leibovich 1983, Thorpe 2004) in which the drift relaxes to the classical Stokes drift (Stokes 1847). In surface layers, on the other hand, the drift can be vastly different to the Stokes drift (Phillips et al. 2010).

Different too are the free surface boundary conditions necessary to ensure the spanwise wavenumber at onset $l = l_c$ is nonzero, these being constant stress (or Neumann) in deep water (Craik 1977) and mixed or Robin in layers (Cox and Leibovich 1993, Hayes and Phillips 2016). In essence Robin conditions reflect coupling between the perturbed flow and the extra stress it produces while Neumann do not. Hayes and Phillips (2016) investigated the role played by this extra stress and found it acts to diminish the growth rate, such that if the growth rate at $l = 0$ is zero with Neumann conditions, then Robin conditions render it negative.

In order to isolate the importance of the extra stress, Hayes and Phillips (2016) chose to proceed analytically. To that end they follow Cox and Leibovich (1993), who first addressed the boundary condition conundrum and noted that because $l = 0$ is of interest, it is sensible to consider events in the limit $l \to 0$. To wit, to expand dependent variables in terms of the small parameter $l$. Cox and Leibovich (1993) further assume the mean shear $U'$ and differential drift $D'$ are uniform (here prime denotes $d/dz$), whereas Hayes and Phillips (2016) leave $U'$ and $D'$ as arbitrary functions that can be set to actual distributions given by Phillips et al. (2010). Both studies impose Robin conditions on top and bottom of the layer to realize, what is in essence, a study of long waves on a sheared surface layer bounded below by a strong thermocline.

Our object in the present work is to employ this framework in the weakly nonlinear regime. By doing so we can ascertain not only the role nonlinearities play on aspect ratio but also how both it and onset to instability are influenced by the extra stress reflected in the boundary conditions.

LC have, of course, been investigated in the nonlinear regime before. On the analytical side, Cox and Leibovich (1993) derive an evolution equation with a cubic nonlinearity that assumes a long wave approximation and show that it is susceptible to symmetry breaking, while Chini (2008) assumes Neumann boundary conditions and considers strongly nonlinear LC both asymptotically and numerically. Chini's (2008) asymptotics do not predict a preferred spanwise wavenumber, but under strong supercritical forcing and on reaching a saturated state, his numerics do. In explanation he notes that wavenumber selection can result from large-scale modulation of nonlinear modes (Newell et al. 1990, Phillips 2015) causing the numerics, which comprise a discrete spectrum of modes, to defer to a finite rather than zero wavenumber. In fact supercritical forcing alone is sufficient to ensure the least stable wavenumber is finite with Neumann conditions, even in the absence of nonlinearities (Hayes and Phillips 2016). Nevertheless, a consistent initial value problem describing the aetiology of LC must predict a preferred nonzero spanwise wavenumber at onset, a requirement not ensured by Neumann boundary conditions. Robin boundary conditions, on the other hand, provide that assurance.

We begin in section 2 by stating the problem and in section 3 outline Hayes and Phillips' (2016) linear perturbation solution. We go on in section 4 to develop a nonlinear perturbation solution and then in section 5 explore the nonlinear steady states. A complimentary numerical solution is outlined in section 6 and results are given in section 7. We find that our expansion is valid for significantly larger $l$ than the $l \to 0$ limit for which it is designed and that nonlinearities play an increasingly important role as $l$ increases. Specifically, nonlinearities are found to suppress instability, indicating supercritical stability. We further find that the nonlinear steady states depict onset spacings almost twice their linear counterparts. Our findings are discussed in section 8.

---

## 2. Governing Equations

### 2.1 Background

We set $z$ vertical (positive upwards), $y$ cross-stream and question the stability of a unidirectional mean shear flow aligned in the $x$ direction in the presence of a field of neutral surface waves that ride on the sheared flow and likewise travel in $x$. The mean flow is denoted by $U(z)$, component velocities by $\mathbf{u} = (u, v, w)$ in $(x, y, z)$ and the drift by $\mathbf{d} = (D(y, z), 0, 0)$, although in order to explore the CL2 instability we restrict attention to the case $D = D(z)$, which for instability requires $D' U' > 0$ (Leibovich 1983). In this instance the force term $\mathbf{d} \times \nabla \times \mathbf{u}$ in the mean field CL-equations then reduces, after cross differentiation, to $D' u_y$ as seen in (1c).

Aside from mentioning that length is scaled by the water depth $h$, so that $z \in [-1, 0]$, with $z = 0$ at the mean free surface we will not dwell on detailed nondimensionalization which is given elsewhere (see e.g. Cox and Leibovich (1993), Phillips and Dai (2014)). All other variables are dimensionless with respect to the mean velocity at the free surface $U$, an eddy viscosity representative of the turbulent diffusivity of momentum $\nu_T$ and the maximum value of the drift $D$, so that in the absence of stratification the evolution equations retain only one parameter, a Rayleigh number:

$$Ra = \frac{U D h^2}{\nu_T^2} \tag{Ra-def}$$

That said, the time $h^2 \nu_T^{-1}$ and velocity scales of the instability are directly affected by the ratio of the characteristic velocity of the mean flow to the wave phase velocity. That ratio in waves of characteristic slope $\epsilon \ll 1$ is $O(\epsilon^s)$ (Craik 1982), where $s \in [0, 2]$ is the shear index (Phillips 1998a). For reference, $s = 2$ in the open ocean (Craik and Leibovich 1976), $s = 1$ in shallow coastal layers (Phillips and Dai 2014) and $s = 0$ in laboratory experiments (Melville et al. 1998). In developing the CL-equations, Craik and Leibovich (1976) thus had the $s = 2$ case in mind, for which $t$ scales as $O(\epsilon^2)$ and velocities scale as $[\epsilon^2(U + u), \epsilon^2 v, \epsilon^2 w]$.

Herein our focus is with the $s = 1$ case, where in contrast $t$ scales as $O(\epsilon^{3/2})$ while velocities scale as $[\epsilon(U + u), \epsilon^2 v, \epsilon^2 w]$ (Phillips 1998a). Of note is that the streamwise component of velocity is $O(\epsilon)$ stronger than its cross-stream counterparts and of concern is whether it is of sufficient strength to modulate the wave field.

Such modulation is ubiquitous when $s = 0$ (Craik 1982, Phillips 1998a) and its action is seen to profoundly affect instability (Phillips and Wu 1994, Phillips and Shen 1996, Phillips 2005). To consider this further we turn to the CLg-equations which accommodate all levels of $s$ and find in fact that modulation plays no role in the interior when $s = 1$ (Phillips and Dai 2014). This result simplifies our analysis on two fronts: first because the imposed wave field must then remain neutral, which means wave amplitude can be excluded as a variable in our weakly nonlinear analysis. And second because, in spite of scaling differences, the CLg-equations contract to the CL-equations for both $s = 2$ and $s = 1$, albeit with differences in scaling.

The planar CL-equations describing the perturbed motion over $z \in [-1, 0]$ in an unstratified fluid are (Leibovich and Paolucci 1981, Phillips 2001a):

$$U_\tau - U_{zz} = -G \tag{1a}$$

$$u_t - \nabla^2 u = U' \psi_y + J(\psi, u) \tag{1b}$$

$$(\partial_t - \nabla^2) \nabla^2 \psi = Ra \, D' u_y + J(\psi, \nabla^2 \psi) \tag{1c}$$

where from mass conservation $v = \psi_z$ and $w = -\psi_y$, while $\nabla^2$ is the planar Laplacian and the Jacobian $J(a, b) = a_y b_z - a_z b_y$.

Herein $U$ and $(u, v, w)$ evolve on disparate time scales $\tau$ and $t$ (as discussed in Phillips and Dai (2014)), while $G$ is a mean streamwise body force (Phillips et al. 2010). Our task is to seek non-decaying solutions to the set (1) for $(U + u, v, w)$ subject to appropriate initial and boundary conditions.

### 2.2 Boundary Conditions

Phillips and Dai (2014) formally derive free-surface boundary conditions for the shallow layer $s = 1$ case. In order to understand their procedure it is important to recognize first, that three disparate time scales for the waves, $T_w$, shear $T_s$ and Langmuir circulation $T_{LC}$ play a role and second, that the set (1), to which the boundary conditions apply, are mean field equations averaged over the smallest time scale $T_w$. Averaging does not, of course, alter the dynamical boundary conditions that apply to viscous flows, namely that the normal and tangential stresses are continuous at the free surface. But what averaging does do in multiscale problems such as this, is to distribute the stresses over the various scales. To wit, any imposed tangential stress is reflected at time scale $T_s \gg T_w$ and absorbed solely by the mean shear $U'$, while any extra tangential stress produced by coupling between the perturbed motion (LC) is reflected at time scale $T_{LC} \gg T_w$. On the other hand, the normal stress is reflected solely by the perturbed flow and thus acts at time scale $T_{LC}$. Phillips and Dai (2014) invoke no averaging when deriving the boundary conditions, but instead decompose the velocity field into components that reflect the three time scales and allow the aforementioned parsing at each timescale to emerge in the analysis. Specifically, they require the velocity field to satisfy the Navier–Stokes equations subject to the requirements of continuity of pressure at the interface and that the interface be a material surface. They further allow for the wave field to be rotational. Of key importance is that their analysis indicates the boundary conditions are mixed and that the extra stress depicted therein is affected by details of the wave field.

Of particular interest here, however, is what happens in the limit as the extra stress goes to zero, in which instance the boundary conditions reduce to Neumann. Since details of the wave field are likely secondary in this context, we here assume the extra stress is constant. The boundary conditions then reduce to the simple mixed Robin form introduced by Cox and Leibovich (1993):

**At $z = 0$ (free surface):**

$$U' = \varsigma \tag{2a}$$

$$u_z + \gamma_s u = 0 \tag{2b}$$

$$\psi_{zz} + \frac{\gamma_s}{2} \psi_z = 0 \tag{2c}$$

$$\psi = 0 \tag{2d}$$

**At $z = -1$ (bottom):**

$$U = 0 \tag{3a}$$

$$-u_z + \gamma_b u = 0 \tag{3b}$$

$$-\psi_{zz} + \gamma_b \psi_z = 0 \tag{3c}$$

$$\psi = 0 \tag{3d}$$

Here $\varsigma$ in (2a) is an imposed stress that acts over time scale $T_s$ and applies to (1a) as does (3a), while (2b–d) and (3b–d) act over $T_{LC}$ and apply to (1b,c). The $\gamma$'s are non-negative parameters which, when set to zero, ensure (2) and (3) reduce to Neumann boundary conditions; $\gamma_s$ acts at the free surface and $\gamma_b$ at the bottom. Phillips and Dai (2014) find that not only are the parameters functions of the spanwise and streamwise wavenumbers but that they are different in the $u$ and $\psi$ equations. Cox and Leibovich (1993), on the other hand, assume the parameters are constants and use physical arguments to estimate them, finding $\gamma_s \approx 0.06$ and $\gamma_b \approx 0.28$.

---

## 3. Linear Perturbation Solution

Cox and Leibovich (1993) and Hayes and Phillips (2016) used the linearized form of (1) subject to boundary conditions (2), (3) to study the stability of the basic sheared state $U'$ in the presence of an aligned differential drift field $D'$. Unlike Cox and Leibovich (1993), however, who set $D' = U' = 1$, Hayes and Phillips (2016) allow $D'$ and $U'$ to each be functions of $z$, in accord with the findings of Phillips et al. (2010). Further, since Phillips and Dai (2014) show that the mean flow evolves on a longer time scale than LC, Hayes and Phillips (2016) chose to treat $U$ as though it were steady at a particular value of $\tau$. They then consider disturbances about it as normal modes proportional to $e^{ily + \sigma t}$, where $l$ is the spanwise wavenumber and $\sigma$ is the growth rate. Disturbances are thus damped and the basic state is stable when $\text{Re}\{\sigma\} < 0$ and grow when $\text{Re}\{\sigma\} > 0$, the transition from one state to the other occurring as the parameter $Ra$ increases through a threshold value, the minimal of which is denoted the critical Rayleigh number, which occurs at the critical wavenumber $l_c$.

Both studies sought to proceed as far as possible analytically and noted that because $l_c = O(\gamma^{1/4})$ for the class of equations (1) when $\gamma \ll 1$ (Chapman and Proctor 1980, Gertsberg and Sivashinsky 1981), a fruitful path forward is via a small-$l$ expansion for the linear growth of small disturbances. On noting that $u$ and $\psi$ are out of phase in $y$, they then write:

$$u(y, z, t) = e^{ily} e^{\sigma t} \sum_{k=0}^{\infty} l^{2k} u_{2k} \tag{4a}$$

$$\psi(y, z, t) = i \, e^{ily} e^{\sigma t} \sum_{k=0}^{\infty} l^{2k+1} \psi_{2k+1} \tag{4b}$$

where $u_{2k}$ and $\psi_{2k+1}$ are functions solely of $z$, with

$$\sigma = \sum_{k=0}^{\infty} l^{2k+2} \sigma_{2k+2} \tag{5a}$$

$$Ra = R = \sum_{k=0}^{\infty} l^{2k} R_{2k} \tag{5b}$$

and, because $l_c = O(\gamma^{1/4})$, introduce the scaling:

$$(\gamma_s, \gamma_b) = l^4 (\tilde{\gamma}_s, \tilde{\gamma}_b) \quad \text{with} \quad \gamma = \gamma_s + \gamma_b \tag{6}$$

Of course $U(z)$ must remain a solution to (1a), but to ensure progress analytically, Hayes and Phillips (2016) approximate Phillips et al.'s (2010) exact solutions for $U'$ and $D'$ with polynomials of degree $N$ and $M$ respectively, as:

$$D' = \sum_{m=0}^{M} a_m z^m \tag{7a}$$

$$U' = \sum_{n=0}^{N} b_n z^n \tag{7b}$$

Here $a_m$ and $b_n$ are constant coefficients, such that $U' = D' = 1$ is recovered when $a_0 = b_0 = 1$ and $M = N = 0$.

Hayes and Phillips (2016) craft an algorithm that allows the calculation to proceed to $O(l^P)$, for any $P \ge 0$. Their path is to substitute the expansions (4) to (6) into the evolution equations (1) and the boundary conditions (2), (3), utilize the Cauchy product formula (Hardy 1949):

$$\left(\sum_{m=0}^{\infty} A_m x^m\right) \left(\sum_{n=0}^{\infty} B_n x^n\right) = \sum_{m=0}^{\infty} \left(\sum_{n=0}^{m} A_{m-n} B_n\right) x^m \tag{8}$$

to equate like powers of $l$ and solve the resulting equations at successive orders in $l$. We outline their general case in section 3.1 and illustrate the process in section 3.2 by calculating several results required later in our nonlinear expansion. The algorithm necessarily recovers the results of Cox and Leibovich (1993), who proceeded to $O(l^4)$.

### 3.1 The General Case to $O(l^P)$

Subject to the interpretation that variables with negative subscripts are zero, Hayes and Phillips (2016) find at $O(l^{2j})$ for $j \ge 0$ that:

$$u''_{2j} - u_{2j-2} - U' \psi_{2j-1} - \sum_{m=0}^{j-1} \sigma_{2m+2} u_{2j-2m-2} = 0 \tag{9}$$

with boundary conditions:

$$u'_{2j} + \tilde{\gamma}_s u_{2j-4} = 0 \quad \text{on } z = 0 \tag{10}$$

$$u'_{2j} - \tilde{\gamma}_b u_{2j-4} = 0 \quad \text{on } z = -1 \tag{11}$$

Their task then is to expose the growth rate $\sigma_{2m+2}$ and realize a solvability condition. For that it is necessary to integrate (9) with respect to $z$ from $-1$ to $0$ and apply the boundary conditions (10), (11). Then, on choosing the resulting constants of integration to ensure zero net mass flux at each order, namely:

$$\int_{-1}^{0} u_{2j} \, dz = \delta_{0,j} = \begin{cases} 1, & j = 0 \\ 0, & j \ne 0 \end{cases} \tag{12}$$

necessitates that the summation term in (9) contracts to $\sigma_{2j}$ as:

$$\sum_{m=0}^{j-1} \sigma_{2m+2} \int_{-1}^{0} u_{2j-2m-2} \, dz = \sigma_{2j}$$

yielding the solvability condition:

$$-\sigma_{2j} = \delta_{0,j-1} + \tilde{\gamma}_s u_{2j-4}\big|_{z=0} + \tilde{\gamma}_b u_{2j-4}\big|_{z=-1} + \int_{-1}^{0} U' \psi_{2j-1} \, dz \tag{13}$$

a result we will use in section 3.2. Also of interest, of course, are $u$ and $\psi$, so on integrating (9) twice with respect to $z$ and setting $\sigma_{2m+2} = 0$, yields at onset to instability that:

$$u_{2j} = \iint^z u_{2j-2} \, dz^2 + \iint^z U' \psi_{2j-1} \, dz^2 + c_{0,j} z + c_{1,j} \tag{14}$$

where the $c_{i,j}$'s are provided in Appendix A.

An odd order in $l$ is necessary to determine $\psi$, where at $O(l^{2j+1})$ Hayes and Phillips (2016) find:

$$\psi''''_{2j+1} = 2 \psi''_{2j-1} - \psi_{2j-3} - \sum_{m=0}^{j} R_{2m} D' u_{2j-2m} + \sum_{m=0}^{j-1} \sigma_{2m+2} \psi''_{2j-2m-1} - \sum_{m=0}^{j-2} \sigma_{2m+2} \psi_{2j-2m-3} \tag{15}$$

with the boundary conditions:

$$\psi''_{2j+1} + \frac{\tilde{\gamma}_s}{2} \psi'_{2j-3} = \psi_{2j+1} = 0 \quad \text{on } z = 0 \tag{16}$$

$$\psi''_{2j+1} - \tilde{\gamma}_b \psi'_{2j-3} = \psi_{2j+1} = 0 \quad \text{on } z = -1 \tag{17}$$

Integration of (15) at onset to instability then leads to:

$$\psi_{2j+1} = 2 \iint^z \psi_{2j-1} \, dz^2 - \iiiint^z \psi_{2j-3} \, dz^4 - \sum_{m=0}^{j} R_{2m} \iiiint^z D' u_{2j-2m} \, dz^4 + \frac{c_{2,j}}{6} z^3 + \frac{c_{3,j}}{2} z^2 + c_{4,j} z + c_{5,j} \tag{18}$$

which may be expressed as:

$$\psi_{2j+1} = \hat{\psi}_{2j+1} - R_{2j} \tilde{\psi}_{2j+1} \tag{19}$$

in which $\hat{\psi}_{2j+1}$ and $\tilde{\psi}_{2j+1}$ are devoid of $R_{2j}$. On using (7) to specify $D'$ and $U'$, then allows $u_{2j}$ and $\psi_{2j+1}$ to be computed to any desired order using computational algebra.

Finally, on substituting (19) into (13) and noting that $R_{2j-2}$ is synonymous with $\sigma_{2j} = 0$ at successive $j$, yields:

$$R_{2j-2} = \frac{\tilde{\gamma}_s u_{2j-4}\big|_{z=0} + \tilde{\gamma}_b u_{2j-4}\big|_{z=-1} + \int_{-1}^{0} \hat{\psi}_{2j-1} U' \, dz + \delta_{0,j-1}}{\int_{-1}^{0} \tilde{\psi}_{2j-1} U' \, dz} \tag{20}$$

from which the neutral curve $R(l)$ follows from (5b).

### 3.2 Some Specific Examples

Several linear results are required in our nonlinear calculation and we highlight them here. First, in view of (9) with boundary conditions (10) and (11) at $O(l^0)$, we may, without affecting $R$ or $\sigma$, set:

$$u_0 = 1 \tag{21}$$

Moreover from (15) we find at $O(l)$ that:

$$\psi''''_1 = -D' R_0 \tag{22}$$

with boundary conditions:

$$\psi''_1 = \psi_1 = 0 \quad \text{on } z = 0, -1 \tag{23}$$

so:

$$\psi_1 = -R_0 \iiiint^z D' \, dz^4 + \frac{c_{2,0}}{6} z^3 + \frac{c_{3,0}}{2} z^2 + c_{4,0} z + c_{5,0} \tag{24}$$

Finally, in order to isolate $R_0$, we let $\psi_1 = -R_0 \tilde{\psi}_1$ and note from (13) at $O(l^2)$ that:

$$-\sigma_2 = 1 - R_0 \int_{-1}^{0} \tilde{\psi}_1 U' \, dz \tag{25}$$

Then because $R_0$ and $\sigma_2 = 0$ are synonymous (Nield 1967), we require from (25) or more generally from (20) that:

$$R_0^{-1} = \int_{-1}^{0} \tilde{\psi}_1 U' \, dz \tag{26}$$

---

## 4. Nonlinear Perturbation Solution

Hayes and Phillips (2016) found that their linear solutions remained valid well beyond the $l \ll 1$ range for which they were derived; indeed, in all cases they considered, the asymptotic solution to $O(l^6)$ remained valid to $l = O(1)$. Our intent now is to build on our linearized solution and craft a small-$l$ perturbation solution to the nonlinear CL-equations (1) on the presumption (later justified) that say at $O(l^6)$ it too remains valid to $l / l_{c_{NL}} = O(1)$.

To that end, and because the linearized solutions at leading order go as $u \sim e^{ily} e^{\sigma_2 l^2 t} u_0$ and $\psi \sim i e^{ily} e^{\sigma_2 l^2 t} l \psi_1$, we introduce the scalings:

$$Y = ly, \quad T = l^2 t \tag{27}$$

along with:

$$\tilde{u}(Y, z, T) = u(y, z, t) \quad \text{and} \quad l \tilde{\Psi}(Y, z, T) = \psi(y, z, t) \tag{28}$$

With this rescaling (1b) becomes:

$$l^2 \frac{\partial \tilde{u}}{\partial T} - l^2 \frac{\partial^2 \tilde{u}}{\partial Y^2} - \frac{\partial^2 \tilde{u}}{\partial z^2} = l^2 U' \frac{\partial \tilde{\Psi}}{\partial Y} + l^2 \frac{\partial \tilde{\Psi}}{\partial Y} \frac{\partial \tilde{u}}{\partial z} - l^2 \frac{\partial \tilde{\Psi}}{\partial z} \frac{\partial \tilde{u}}{\partial Y} \tag{29}$$

while (1c) becomes:

$$l^2 \frac{\partial}{\partial T}\left(l^2 \frac{\partial^2 \tilde{\Psi}}{\partial Y^2} + \frac{\partial^2 \tilde{\Psi}}{\partial z^2}\right) - l^4 \frac{\partial^4 \tilde{\Psi}}{\partial Y^4} - 2 l^2 \frac{\partial^4 \tilde{\Psi}}{\partial Y^2 \partial z^2} - \frac{\partial^4 \tilde{\Psi}}{\partial z^4}$$
$$= Ra \, D' \frac{\partial \tilde{u}}{\partial Y} + l^2 \frac{\partial \tilde{\Psi}}{\partial Y} \frac{\partial}{\partial z}\left(l^2 \frac{\partial^2 \tilde{\Psi}}{\partial Y^2} + \frac{\partial^2 \tilde{\Psi}}{\partial z^2}\right) - l^2 \frac{\partial \tilde{\Psi}}{\partial z} \frac{\partial}{\partial Y}\left(l^2 \frac{\partial^2 \tilde{\Psi}}{\partial Y^2} + \frac{\partial^2 \tilde{\Psi}}{\partial z^2}\right) \tag{30}$$

Accordingly the boundary conditions (2) and (3) become:

**At $z = 0$:**

$$\frac{\partial^2 \tilde{\Psi}}{\partial z^2} + \frac{1}{2} l^4 \tilde{\gamma}_s \frac{\partial \tilde{\Psi}}{\partial z} = \tilde{\Psi} = \frac{\partial \tilde{u}}{\partial z} + l^4 \tilde{\gamma}_s \tilde{u} = 0 \tag{31}$$

**At $z = -1$:**

$$\frac{\partial^2 \tilde{\Psi}}{\partial z^2} - l^4 \tilde{\gamma}_b \frac{\partial \tilde{\Psi}}{\partial z} = \tilde{\Psi} = \frac{\partial \tilde{u}}{\partial z} - l^4 \tilde{\gamma}_b \tilde{u} = 0 \tag{32}$$

As before we seek an expansion in $l$ and so, in accord with (4) and (5), expand as:

$$\tilde{u} = \sum_{k=0}^{\infty} l^{2k} u_{2k}(Y, z, T) \tag{33a}$$

$$\tilde{\Psi} = \sum_{k=0}^{\infty} l^{2k} \Psi_{2k}(Y, z, T) \tag{33b}$$

with:

$$Ra = R = \sum_{k=0}^{\infty} l^{2k} R_{2k} \tag{34}$$

On substituting (33) and (34) into (29) to (32) and equating like powers of $l$ using the Cauchy product formula (8), we then have at $O(l^{2k})$, again with the interpretation that variables with negative subscripts are zero, that:

$$\frac{\partial u_{2(k-1)}}{\partial T} - \frac{\partial^2 u_{2(k-1)}}{\partial Y^2} - \frac{\partial^2 u_{2k}}{\partial z^2} = U' \frac{\partial \Psi_{2(k-1)}}{\partial Y} + \sum_{m=0}^{k-1} \left(\frac{\partial \Psi_{2(k-m-1)}}{\partial Y} \frac{\partial u_{2m}}{\partial z} - \frac{\partial \Psi_{2(k-m-1)}}{\partial z} \frac{\partial u_{2m}}{\partial Y}\right) \tag{35}$$

with boundary conditions:

$$\frac{\partial u_{2k}}{\partial z} + \tilde{\gamma}_s u_{2(k-2)} = 0 \quad \text{on } z = 0 \tag{36}$$

$$\frac{\partial u_{2k}}{\partial z} - \tilde{\gamma}_b u_{2(k-2)} = 0 \quad \text{on } z = -1 \tag{37}$$

Accordingly at $O(l^{2k+1})$ we have:

$$\frac{\partial^3 \Psi_{2(k-2)}}{\partial T \partial Y^2} + \frac{\partial^3 \Psi_{2(k-1)}}{\partial T \partial z^2} - \frac{\partial^4 \Psi_{2(k-2)}}{\partial Y^4} - 2 \frac{\partial^4 \Psi_{2(k-1)}}{\partial Y^2 \partial z^2} - \frac{\partial^4 \Psi_{2k}}{\partial z^4}$$
$$= \sum_{m=0}^{k} R_{2(k-m)} D' \frac{\partial u_{2m}}{\partial Y} + \sum_{m=0}^{k-1} \left(\frac{\partial \Psi_{2(k-m-1)}}{\partial Y} \frac{\partial^3 \Psi_{2m}}{\partial z^3} - \frac{\partial \Psi_{2(k-m-1)}}{\partial z} \frac{\partial^3 \Psi_{2m}}{\partial Y \partial z^2}\right)$$
$$- \sum_{m=0}^{k-2} \left(\frac{\partial \Psi_{2(k-m-2)}}{\partial z} \frac{\partial^3 \Psi_{2m}}{\partial Y^3} - \frac{\partial \Psi_{2(k-m-2)}}{\partial Y} \frac{\partial^3 \Psi_{2m}}{\partial z \partial Y^2}\right) \tag{38}$$

with boundary conditions:

$$\frac{\partial^2 \Psi_{2k}}{\partial z^2} + \frac{1}{2} \tilde{\gamma}_s \frac{\partial \Psi_{2(k-2)}}{\partial z} = \Psi_{2k} = 0 \quad \text{on } z = 0 \tag{39}$$

$$\frac{\partial^2 \Psi_{2k}}{\partial z^2} - \tilde{\gamma}_b \frac{\partial \Psi_{2(k-2)}}{\partial z} = \Psi_{2k} = 0 \quad \text{on } z = -1 \tag{40}$$

### 4.1 The First Few Orders

Equations (35) to (40) must be solved for successive integers $k \ge 0$. To outline the process we provide details up to $O(l^3)$, noting that to this order the boundary conditions reduce to Neumann conditions.

**At $O(l^0)$:**

$$\frac{\partial^2 u_0}{\partial z^2} = 0 \tag{41}$$

with boundary conditions:

$$\frac{\partial u_0}{\partial z} = 0 \quad \text{on } z = 0, -1 \tag{42}$$

Thus:

$$u_0 = u_0(Y, T) \tag{43}$$

where for the moment the form of the arbitrary function $u_0(Y, T)$ is undetermined.

**At $O(l)$:**

$$\frac{\partial^4 \Psi_0}{\partial z^4} = -R_0 D' \frac{\partial u_0}{\partial Y} \tag{44}$$

with boundary conditions:

$$\frac{\partial^2 \Psi_0}{\partial z^2} = \Psi_0 = 0 \quad \text{on } z = 0, -1 \tag{45}$$

Since (44) is readily integrable the solution is simply:

$$\Psi_0 = -R_0 \tilde{\psi}_1 \frac{\partial u_0}{\partial Y} = -R_0 \tilde{\Psi}_0 \tag{46}$$

in which $\tilde{\psi}_1$ is known from the linear problem (24).

**Nonlinearities are first evident at $O(l^2)$:**

$$\frac{\partial^2 u_2}{\partial z^2} = \frac{\partial u_0}{\partial T} - \frac{\partial^2 u_0}{\partial Y^2} - \frac{\partial \Psi_0}{\partial Y} U' + \frac{\partial \Psi_0}{\partial z} \frac{\partial u_0}{\partial Y} - \frac{\partial \Psi_0}{\partial Y} \frac{\partial u_0}{\partial z} \tag{47}$$

with boundary conditions:

$$\frac{\partial u_2}{\partial z} = 0 \quad \text{on } z = 0, -1 \tag{48}$$

Of course equation (47) defines $u_2$, but because $u_0$ is recurrent in it we can exploit it to extract a general solution for $u_0$. Specifically on integrating (47) with respect to $z$ from $-1$ to $0$ and applying the boundary conditions (45) and (48) to expunge $u_2$, we find that $u_0$ must satisfy:

$$\frac{\partial u_0}{\partial T} - \frac{\partial^2 u_0}{\partial Y^2} = \int_{-1}^{0} \frac{\partial \Psi_0}{\partial Y} U' \, dz \tag{49}$$

On substituting (46) for $\Psi_0$, we then obtain:

$$\frac{\partial u_0}{\partial T} - \frac{\partial^2 u_0}{\partial Y^2} \left(1 - R_0 \int_{-1}^{0} \tilde{\psi}_1 U' \, dz\right) = 0 \tag{50}$$

from which it is evident from (26) that $R_0 = R_0$ and from (25) that (50) can be written in terms of the linear growth rate $\sigma_2$, as:

$$\frac{\partial u_0}{\partial T} + \sigma_2 \frac{\partial^2 u_0}{\partial Y^2} = 0 \tag{51}$$

Thus to this order the nonlinear steady states $\partial / \partial T = 0$ are synonymous with linear neutral stability, $\sigma_2 = 0$.

Equation (51) is readily solved either by similarity methods or by the method of separation of variables. Since we seek solutions periodic in $Y$, however, we choose the latter for which, in view of (21):

$$u_0 = \sum_{m=0}^{\infty} h_m e^{\sigma_2 m^2 T} \cos(mY) \tag{52}$$

It then follows from (46) that $\Psi_0$ is:

$$\Psi_0 = R_0 \tilde{\psi}_1 \sum_{m=0}^{\infty} m \, h_m e^{\sigma_2 m^2 T} \sin(mY) \tag{53}$$

where $h_m$ are constant coefficients.

Returning now to $u_2$, we find on integrating (47) twice with respect to $z$ that:

$$u_2 = -\iint^z \frac{\partial \Psi_0}{\partial Y} U' \, dz^2 + \iint^z \frac{\partial \Psi_0}{\partial z} \frac{\partial u_0}{\partial Y} \, dz^2 + \left(\frac{\partial u_0}{\partial T} - \frac{\partial^2 u_0}{\partial Y^2}\right) \frac{z^2}{2} + z \int^z \frac{\partial \Psi_0}{\partial Y} U' \, dz\bigg|_{z=0} + g(Y, T) \tag{54}$$

where $g$ is an arbitrary function of $Y$ and $T$. We can likewise deduce $\Psi_2$, where at $O(l^3)$:

$$\frac{\partial^3 \Psi_0}{\partial T \partial z^2} - 2 \frac{\partial^4 \Psi_0}{\partial z^2 \partial Y^2} - \frac{\partial^4 \Psi_2}{\partial z^4} - R_2 D' \frac{\partial u_0}{\partial Y} - R_0 D' \frac{\partial u_2}{\partial Y} - \frac{\partial \Psi_0}{\partial Y} \frac{\partial^3 \Psi_0}{\partial z^3} + \frac{\partial \Psi_0}{\partial z} \frac{\partial^3 \Psi_0}{\partial z^2 \partial Y} = 0 \tag{55}$$

with boundary conditions:

$$\frac{\partial^2 \Psi_2}{\partial z^2} = \Psi_2 = 0 \quad \text{on } z = 0, -1 \tag{56}$$

Solving gives:

$$\Psi_2 = \iint^z \frac{\partial \Psi_0}{\partial T} \, dz^2 - 2 \iint^z \frac{\partial^2 \Psi_0}{\partial Y^2} \, dz^2 - R_2 \frac{\partial u_0}{\partial Y} \iiiint^z D' \, dz^4 - R_0 \iiiint^z D' \frac{\partial u_2}{\partial Y} \, dz^4$$
$$- \iiiint^z \frac{\partial \Psi_0}{\partial Y} \frac{\partial^3 \Psi_0}{\partial z^3} \, dz^4 + \iiiint^z \frac{\partial \Psi_0}{\partial z} \frac{\partial^3 \Psi_0}{\partial Y \partial z^2} \, dz^4 + \frac{f_0}{6} z^3 + \frac{f_1}{2} z^2 + f_2 z + f_3 \tag{57}$$

where the $f_i$'s are given in Appendix B. Furthermore, in line with (19), we may recast (57) and higher orders as:

$$\Psi_{2j} = \hat{\Psi}_{2j} - R_{2j} \tilde{\Psi}_{2j} \tag{58}$$

where $\hat{\Psi}_{2j}$ and $\tilde{\Psi}_{2j}$ are devoid of $R_{2j}$.

Finally, in view of (52) and (53), it is clear that $u_2$ and $\Psi_2$, and subsequently $u_{2k}$ and $\Psi_{2k}$ to all orders, comprise a discrete spectrum of wavenumbers in $Y = ly$, so it is intuitive to expand each as:

$$u_{2k}(Y, z, T) = \sum_{m=0}^{\infty} u_{2k,m}(z, T) \cos(mY) \tag{59}$$

$$\Psi_{2k}(Y, z, T) = \sum_{m=0}^{\infty} \Psi_{2k,m}(z, T) \sin(mY) \tag{60}$$

On doing so and after substituting (59) and (60) into (35) to (40), harmonics in $Y$ result from the nonlinear terms. Of key importance is that terms of order higher than the expansion of the solution be discarded in accordance with the theorem given in Appendix C. The resulting equations for $u_{2k,m}$ and $\Psi_{2k,m}$ are then solved at successive orders in $l$. By proceeding this way arbitrary constants of integration appear in the nonlinear perturbation solution analogous to those in the linear one. In seeking nonlinear steady states, these were chosen to ensure $u_{2k,m}\big|_{z=0} = \delta_{2k,0} \delta_{m,1}$.

---

## 5. Nonlinear Steady States

We saw in section 4 that integration of (47) at $O(l^2)$ and application of the boundary conditions (45) and (48) lead to an expression solely in terms of $u_0$. In fact if we apply the same process at any order we obtain from (35) that:

$$\int_{-1}^{0} \frac{\partial u_{2(k-1)}}{\partial T} \, dz = -\tilde{\gamma}_s u_{2(k-2)}\bigg|_{z=0} - \tilde{\gamma}_b u_{2(k-2)}\bigg|_{z=-1} + \int_{-1}^{0} \frac{\partial^2 u_{2(k-1)}}{\partial Y^2} \, dz$$
$$+ \int_{-1}^{0} U' \frac{\partial \Psi_{2(k-1)}}{\partial Y} \, dz + \sum_{m=0}^{k-1} \int_{-1}^{0} \left(\frac{\partial \Psi_{2(k-m-1)}}{\partial Y} \frac{\partial u_{2m}}{\partial z} - \frac{\partial \Psi_{2(k-m-1)}}{\partial z} \frac{\partial u_{2m}}{\partial Y}\right) dz \tag{61}$$

which contracts to (50) when $k = 1$.

Robin boundary conditions first play a role when $k = 2$, that is $O(l^4)$, where from (61):

$$\int_{-1}^{0} \frac{\partial u_2}{\partial T} \, dz = -\tilde{\gamma} \, u_0 + \int_{-1}^{0} \frac{\partial^2 u_2}{\partial Y^2} \, dz + \int_{-1}^{0} U' \frac{\partial \Psi_2}{\partial Y} \, dz + \int_{-1}^{0} \left(-\frac{\partial \Psi_2}{\partial z} \frac{\partial u_0}{\partial Y} + \frac{\partial \Psi_0}{\partial Y} \frac{\partial u_2}{\partial z} - \frac{\partial \Psi_0}{\partial z} \frac{\partial u_2}{\partial Y}\right) dz \tag{62}$$

Herein we see terms that contain $\Psi_0$ and $\Psi_2$. So since from (46) and (57) where $\Psi_0 = -R_0 \tilde{\Psi}_0$ (in which $R_0$ is known) and $\Psi_2 = \hat{\Psi}_2 - R_2 \tilde{\Psi}_2$, we can set $\partial/\partial T = 0$ and use (62) to evaluate $R_2$ as:

$$R_2 = \frac{\int_{-1}^{0} \left(\frac{\partial \Psi_0}{\partial z} \frac{\partial u_2}{\partial Y} + \frac{\partial \hat{\Psi}_2}{\partial z} \frac{\partial u_0}{\partial Y} - \frac{\partial \Psi_0}{\partial Y} \frac{\partial u_2}{\partial z} - \frac{\partial \hat{\Psi}_2}{\partial Y} U' - \frac{\partial^2 u_2}{\partial Y^2}\right) dz + \tilde{\gamma} \, u_0}{\int_{-1}^{0} \left(-\frac{\partial \tilde{\Psi}_2}{\partial Y} U' + \frac{\partial \tilde{\Psi}_2}{\partial z} \frac{\partial u_0}{\partial Y}\right) dz} \tag{63}$$

Moreover, if we parse (63) into a $\gamma = 0$ portion $R^*_2$ and remaining portion $\tilde{\gamma} \tilde{R}_2$ then we can write:

$$R_2 = R^*_2 + \tilde{\gamma} \tilde{R}_2 \tag{64}$$

and use it with (6) in our expansion (34) for $R$, namely:

$$\bar{R} = R_0 + l^2 R_2 + \cdots = R_0 + l^2 R^*_2 + \frac{\gamma}{l^2} \tilde{R}_2 + \cdots \tag{65}$$

Finally, since critical spacing occurs at $d\bar{R}/dl = 0$, then:

$$l_{c_{NL}} = \left(\frac{\gamma \tilde{R}_2}{R^*_2}\right)^{1/4} \tag{66}$$

in contrast to its linear counterpart (Cox and Leibovich 1993, Hayes and Phillips 2016):

$$l_{c_L} = \left(\frac{\gamma R_0}{R^*_2}\right)^{1/4} \tag{67}$$

which means that the ratio of linear to nonlinear critical spacing:

$$\kappa = \frac{l_{c_L}}{l_{c_{NL}}} = \left(\frac{R_0 R^*_2}{R^*_2 \tilde{R}_2}\right)^{1/4} \tag{68}$$

is independent of $\gamma$. Accordingly, the critical Rayleigh numbers follow as:

$$R_{c_{NL}} = R_0 + 2 \left(\gamma \tilde{R}_2 R^*_2\right)^{1/2} \tag{69}$$

and (Hayes and Phillips 2016):

$$R_{c_L} = R_0 + 2 \left(\gamma R_0 R^*_2\right)^{1/2} \tag{70}$$

---

## 6. Complimentary Numerical Solution

Although the process outlined in sections 4 and 5 is repeatable to any order, the complexity of the expansion becomes progressively more unwieldy as the order increases. Nevertheless it is comfortably handled by computer algebra. That said we do not know whether our expansion is convergent over the range of $l$ that includes $l_c$ and to that end it is expeditious to employ what we have learned and seek a complimentary numerical solution. To that end we solve the primitive equations (1) directly, seeking solutions of a form which separate the functional variation $(y, z, t)$ and capture the spectral aspect depicted in (59) and (60), to wit we write:

$$u = \sum_{m=0}^{J-2} \sum_{k=0}^{I} A_{m,k}(t) P_m(z) \cos(k l y) \tag{71}$$

$$\psi = \sum_{m=0}^{J} \sum_{k=0}^{I} B_{m,k}(t) P_m(z) \sin(k l y) \tag{72}$$

Here the coefficients $A_{m,k}$ and $B_{m,k}$ are unknown functions of $t$ while $P_m(z)$ are the basis functions chosen from a family of orthogonal polynomials that lie on the finite domain; we chose shifted Legendre basis functions on $z \in [-1, 0]$.

After substitution of (71) and (72) into (1) and discarding higher order harmonics, we collect like trigonometrical terms (in accordance with the theorem in Appendix C) to obtain a set of equations in $z$ and $t$ whose residuals we denote $r_{1,i}(z, t)$ and $r_{2,i}(z, t)$. To proceed we require an equation system for $A_{m,k}$ and $B_{m,k}$ devoid of $z$ and thus integrate over $z$ to remove it. More precisely we utilize the Galerkin technique in which we first take the inner product of the residuals and require them to be zero, as:

$$\int_{-1}^{0} r_{1,i} P_j \, dz = \int_{-1}^{0} r_{2,i} P_j \, dz = 0 \tag{73}$$

for $i = 0, 1, \ldots, I$, and $j = 0, 1, \ldots, J-4$. The remaining equations required for closure are obtained by making the same substitution into the boundary conditions (see, e.g. Phillips 2005). This results in a system of nonlinear ordinary differential equations.

In the present work, however, we are specifically interested in the nonlinear steady states and in this instance the system reduces to a set of algebraic equations, which is solved by an adaptive Newton's method. Finally, for consistency with the nonlinear perturbation solution, we specify $A_{0,1}$ to ensure the coefficient of $\cos(ly)$ in $u\big|_{z=0}$ is unity.

---

## 7. Results

We present now results for the nonlinear steady states. Of particular interest is how they compare not only with their linear counterparts, but also with their numerical counterparts. Looking first to the small-$l$ perturbation expansion, the constants $R_{2k}$ for $k = 0, 1, 2, \ldots$ are calculated in the nonlinear analysis and then substituted into (65) to realize $\bar{R}(l)$; the expansion was taken to $O(l^{16})$. Then for comparison with the numerical solution, we set $I = 1$ in our expansions (71) and (72) for $u$ and $\psi$. Of course there is no restriction on $J$, but we found that the accuracy provided by $J = 13$ was sufficient for all cases studied. Results from the expansions and numerics are indistinguishable over a region well in excess of $l_{c_{NL}}$.

### 7.1 Linear Versus Nonlinear

We begin by comparing the nonlinear steady states with the linear neutral curve as $Ra$ vs. $l$ and do so first with Neumann boundary conditions ($\gamma_b = \gamma_s = 0$) in **Figure 1** and then Robin boundary conditions ($\gamma_b = 0$, $\gamma_s = 0.0001$) in **Figure 2** for various combinations of $U'$ and $D'$ as detailed below.

**Figure 1** — Plots of the linear neutral curve and nonlinear steady states as $Ra$ vs. $l$ for **Neumann boundary conditions**:

- (a) $D' = U' = 1$ linear
- (b) $D' = U' = 1$ nonlinear
- (c) $D' = 1, U' = 1 + z$ and $D' = 1 + z, U' = 1$ linear (these coincide)
- (d) $D' = 1 + z, U' = 1$ nonlinear
- (e) $D' = 1, U' = 1 + z$ nonlinear
- (f) $D' = U' = 1 + z$ linear
- (g) $D' = U' = 1 + z$ nonlinear

**Figure 2** — Same as Figure 1 but for **Robin boundary conditions** with $\gamma_b = 0$, $\gamma_s = 0.0001$.

We observe first (in Figures 1 and 2) that the nonlinear steady states appear to asymptote to the neutral curves for the linearized problems when $l \ll 1$ which, if true, would indicate that nonlinearities are small for sufficiently small $l$. To investigate this notion further we plot $\Delta = \bar{R} - R$ against $l$ in **Figures 3** and **4**.

**Figure 3** — Plots of $\Delta = \bar{R} - R$ vs. $l$ for **Neumann boundary conditions**:

- (a) $D' = U' = 1$
- (b) $D' = 1 + z, U' = 1$
- (c) $D' = 1, U' = 1 + z$
- (d) $D' = U' = 1 + z$

Looking first to Figure 3 we do indeed see that $\Delta \to 0$ as $l \to 0$, so that nonlinearities really are small for small $l$. This means that Neumann boundary conditions necessarily predict onset at $l = 0$ irrespective of whether nonlinearities are present or absent, a result in accord with Chapman and Proctor's (1980) finding that nonlinearities do not resolve the $l = 0$ conundrum.

**Figure 4** — Same as Figure 3 but for **Robin boundary conditions** with $\gamma_b = 0$, $\gamma_s = 0.0001$.

In contrast, Figure 4 shows that $\Delta > 0$ at all $l$, indicating that nonlinearities play a role at all $l$ provided $\gamma \ne 0$. This means we have supercritical stability. It further means that Robin boundary conditions ensure finite spacing in both the linear and nonlinear case, as we see in Figure 2.

Also clear in Figures 1 and 2 is that $\bar{R}$ draws away from $R$ as $l$ increases indicating that nonlinearities become increasingly important and, since $\bar{R} > R$, that nonlinearities act to suppress instability. That said, the level of nonlinear suppression is least for the least stable case (b) and increases as $D'$ and $U'$ depart from unity. Further, in contrast to the linear case where the neutral curves for the intermediate cases (c) are identical, the nonlinear steady states (d) and (e) are markedly different, suggesting that the shear profile $U' = 1 + z$ is more effective at suppressing instability than an identical differential drift $D' = 1 + z$ profile.

Turning now to the critical Rayleigh number, it is clear that supercritical stability here requires $R_{c_{NL}} > R_{c_L}$, subject to details of differential drift and shear. We might also expect the critical Rayleigh number to be affected by the degree of coupling between the perturbed motions and the extra stress it produces, that is by the value of $\gamma$. But that is not the case, at least for $\gamma \ll 1$, because from (69) and (70) we find $R_0 \gg 2(\gamma \tilde{R}_2 R^*_2)^{1/2}$ and $R_0 \gg 2(\gamma R_0 R^*_2)^{1/2}$ and thus that $R_{c_{NL}} \approx R_0$ and $R_{c_L} \approx R_0$.

### 7.2 Critical Wavenumber

On the other hand, the critical wavenumber is affected by $\gamma$. Indeed, as we see from (66) and (67), the critical wavenumber grows in both instances as $\gamma^{1/4}$, albeit with different constants of proportionality, as is evident in **Figure 5**, where the baseline linear cases (a) and (b) lie well above the nonlinear cases (c) and (d), to wit $l_{c_{NL}} < l_{c_L}$.

**Figure 5** — Plots of the critical spanwise wavenumber $l_c$ vs. $\gamma_s$ for the linear and nonlinear cases with $\gamma_b = 0$:

- (a) $D' = U' = 1 + z$ linear
- (b) $D' = U' = 1$ and $D' = 1 + z, U' = 1$ and $D' = 1, U' = 1 + z$ linear (these coincide)
- (c) $D' = U' = 1$ and $D' = 1 + z, U' = 1$ nonlinear
- (d) $D' = U' = 1 + z$ and $D' = 1, U' = 1 + z$ nonlinear

That noted, the ratio of linear to nonlinear critical wavenumbers $\kappa$ is independent of $\gamma$ (68). In short, $\kappa$ is a constant for each $U'$, $D'$ pair. For example, for $D' = (1, 1 + z, 1, 1 + z)$ and $U' = (1, 1, 1 + z, 1 + z)$, we find to $O(l^4)$ that:

$$\kappa \approx (1.425, 1.427, 1.919, 1.944)$$

which means that the spacing of LC is up to twice that predicted by linear theory.

This finding is unexpected, because usually the nonlinear spacing saturates at a value close to the linear value, as say in Rayleigh–Bénard convection (Chapman and Proctor 1980). Put differently, if the eigenfunctions underwriting $u$ and $\psi$ retain their linear form as Rayleigh number increases into the supercritical regime, the eigenvalue $l_{c_{NL}}$ too is unaltered. Conversely, if the eigenfunctions change substantially, so too can the spacing; to wit symmetry breaking could spawn a subcritical bifurcation and consequent decrease in spacing.

To clarify whether symmetry breaking does occur we recall that nonlinearities are first evident at $O(l^2)$ and consider (47) which, on noting (43) and using (46), we write as:

$$\frac{\partial^2 u_2}{\partial z^2} = \frac{\partial u_0}{\partial T} - \frac{\partial^2 u_0}{\partial Y^2} + R_0 \left[\tilde{\psi}_1 U' \frac{\partial^2 u_0}{\partial Y^2} - \frac{\partial \tilde{\psi}_1}{\partial z} \left(\frac{\partial u_0}{\partial Y}\right)^2\right] \tag{74}$$

It is immediately clear that (74) is not invariant under the mapping $u_0 \mapsto -u_0$ and that up-down symmetry is broken by the symmetry breaking last term. A subcritical bifurcation, therefore, is not unexpected.

### 7.3 Base Flow Modification

Finally we ask whether the eigenfunctions remain harmonic in $y$ once nonlinearities come into play, or whether they pare into spanwise dependent and spanwise independent components. We note that while the former is typical, the latter is also observed, as say by Hall and Smith (1988) in the context of Taylor–Görtler vortices and by Phillips et al. (1998b) in the context of LC in $s = 0$ shear. Moreover in each instance the correction is of the order of the imposed base flow. To resolve the situation here we write, in accord with (4a,b):

$$u = \sum_{k=0}^{1} u^k_e(z) \cos(kly) \tag{75a}$$

$$\psi = \sum_{k=0}^{1} \Psi^k_e(z) \sin(kly) \tag{75b}$$

and plot the unity normalized eigenmodes $u^k_e$ and $\Psi^k_e$ for the linear and nonlinear cases in **Figure 6**.

**Figure 6** — Plots of the eigenmodes (a) $u^k_e$ and (b) $\Psi^k_e$ at critical linear $R_{c_L} \approx 121.068$, $l_{c_L} \approx 0.150$ and nonlinear $R_{c_{NL}} \approx 122.194$, $l_{c_{NL}} \approx 0.105$ for $D' = U' = 1$ with $\gamma_s = 0.0001$ and $\gamma_b = 0$:

- (i) nonlinear with $k = 0$
- (ii) nonlinear with $k = 1$
- (iii) linear with $k = 1$

Looking first to the $k = 1$ mode we find that although $u^1_e$ has evolved slightly from its linear counterpart, $\Psi^1_e$ is essentially unchanged. But the $y$-independent or $k = 0$ mode, which is necessarily zero in the linear case, is not zero in the nonlinear case, as we see in Figure 6(a) curve (i). This means that the nonlinear interaction of the $k = 1$ modes give rise to a spanwise independent correction $u^0_e(z)$ to the base flow.

---

## 8. Discussion

Shallow water LC arise in lakes, estuaries and coastal waters and typically extend throughout the water column to the floor, where they vastly enhance sediment mixing (Gargett et al. 2004); they are also crucial to the transport of nutrients and plankton (Denman and Gargett 1995, Guasto et al. 2012). Key to understanding conditions under which LC penetrate to the bottom and also in interpreting surface expressions of LC, is their aspect ratio (Marmorino et al. 2005). In deep water, the aspect ratio of LC (width of two cells to depth) is two to three (Smith et al. 1987, Smith 1992); values admirably predicted by linear theory (Leibovich and Paolucci 1981, Phillips 2001a, 2002). But in shallow waters the ratio is circa ten (Marmorino et al. 2005) and is not recovered by linear theory. We should like to know why. Of course the two cases can be quite different. For example, in deep water the shear and drift diminish with depth and have maxima at the surface, while the shear resulting from a tidal current in shallow water has a maxima at the bottom. This shallow water situation vastly alters the drift profile and initiates LC at the bottom (bottom-up) rather than at the surface (top-down) (Phillips et al. 2010). Linear instability theory also suggests it gives rise to highly elongated LC (Phillips and Dai 2014), although their aspect ratio far exceeds those observed.

LC arise through a wave-averaged (that is drift) mean flow interaction and once formed further interact with the wave field to realize at the surface an "extra" mean stress. This extra stress, whose magnitude we represent by $\gamma$, is always present but appears to be important only to shallow water LC, where it acts to suppress instability at small spanwise wavenumbers (Hayes and Phillips 2016) and thereby ensure a preferred spacing at onset (Cox and Leibovich 1993, Phillips and Dai 2014). That said, and while onset details are affected by its magnitude, the extra stress does not explain large aspect LC, at least in linear theory. The question then is whether nonlinearities play a role on aspect ratio and if so, how that role is influenced by shear, drift and the extra stress. Answering that question is the focus of this paper.

First we find (see Figures 1 and 2) that the nonlinear steady states at given spanwise wavenumber occur at a higher Rayleigh number than the neutral curve, which means that nonlinearities act to suppress instability at all wavenumbers. Moreover the level of suppression increases with wavenumber and thereby acts to diminish the range of linearly unstable wavenumbers at a specific Rayleigh number. In other words, we have supercritical stability that acts as a high wavenumber cutoff. Moreover, we find the range of unstable wavenumbers further narrows as the level of shear changes, that is as $U'$ varies from 1 to $1+z$ (Figure 2). A similar variation in $D'$ likewise affects the range of unstable wavenumbers but it is evident that $U'$ plays the stronger role, as we see from curves (d) and (e) in Figure 2. This figure further gives the impression that the nonlinear steady states are asymptotic to the linear neutral curve as wavenumber decreases but that is not so, as we see by plotting their difference in Figure 4. Here we likewise see the dominance of $U'$ over $D'$ and the narrowing of the spectra of unstable wavenumbers with change in $U'$ and $D'$.

These conclusions are not altered by the extra stress at the surface, as we see by comparing Figures 1 with 2, and 3 with 4, but therein we do see the importance of $\gamma$ in ensuring a preferred nonzero wavenumber at onset (Figures 2 and 4). We further find that while the ratio of linear to nonlinear spacing is unaffected by $\gamma$ (68), the value of the onset spacing is affected by $\gamma$; in fact it goes as $\gamma^{1/4}$, as we see in (67) and plot in Figure 5.

Of particular interest in Figure 5 is that nonlinearities noticeably affect the critical wavenumber and indicate that LC aspect ratio is almost double that given by linear theory. Why this occurs is seen in (74) to be the result of a subcritical bifurcation caused by symmetry breaking. This result is the nub of our study, in that nonlinearities can substantially affect LC aspect ratio.

The question remaining, of course, is whether our findings are in concert with observation? To answer that we set $\gamma_s = 0.06$ and $\gamma_b = 0.28$ as suggested by Cox and Leibovich (1993) and calculate the aspect ratio. To proceed, we first note that these values of $\gamma$ imply $l_{c_L} \approx 1.111$ for a layer bounded by a thermocline and $l_{c_L} \approx 1.773$ if the bottom is rigid (Phillips and Dai 2014). Second, since we find that $\kappa \in [1.425, 1.944]$ and that $\kappa$ is independent of $\gamma$, then from (68) $l_{c_{NL}} = l_{c_L} / \kappa$, which means $l_{c_{NL}}$ ranges from 0.57 to 1.24. Finally, since length is normalized by the layer depth then the aspect ratio of two cells to depth $L = 2\pi / l_{c_{NL}}$ ranges from **5 to 11**, which is consistent with the range observed by Marmorino et al. (2005).

---

## Appendix A

### Constants from $u_{2j}$ (equation 14) in the linear perturbation solution algorithm

$$c_{0,j} = -\tilde{\gamma}_s u_{2j-4}\bigg|_{z=0} - \int^z U' \psi_{2j-1} \, dz\bigg|_{z=0} - \int^z u_{2j-2} \, dz\bigg|_{z=0}$$

$$c_{1,j} = -\int_{-1}^{0} \left(\iint^z u_{2j-2} \, dz^2 + \iint^z U' \psi_{2j-1} \, dz^2\right) dz + \delta_{0,j} + \frac{1}{2} c_{0,j}$$

### Constants from $\psi_{2j+1}$ (equation 18) in the perturbation solution algorithm

$$c_{2,j} = \left[-\iint^z \sum_{m=0}^{j} D' u_{2j-2m} R_{2m} \, dz^2 - \iint^z \psi_{2j-3} \, dz^2 + c_{3,j} - \tilde{\gamma}_b \psi'_{2j-3}\right]_{z=-1}$$

$$c_{3,j} = \left[\iint^z \sum_{m=0}^{j} D' u_{2j-2m} R_{2m} \, dz^2 + \iint^z \psi_{2j-3} \, dz^2 - \frac{1}{2} \tilde{\gamma}_s \psi'_{2j-3}\right]_{z=0}$$

$$c_{4,j} = \left[-\iiiint^z \sum_{m=0}^{j} D' u_{2j-2m} R_{2m} \, dz^4 - \iiiint^z \psi_{2j-3} \, dz^4 + 2 \iint^z \psi_{2j-1} \, dz^2\right]_{z=-1} - \frac{1}{6} c_{2,j} + \frac{1}{2} c_{3,j} + c_{5,j}$$

$$c_{5,j} = \left[\iiiint^z \sum_{m=0}^{j} D' u_{2j-2m} R_{2m} \, dz^4 + \iiiint^z \psi_{2j-3} \, dz^4 - 2 \iint^z \psi_{2j-1} \, dz^2\right]_{z=0}$$

---

## Appendix B

### Functions for $\Psi_2$ in equation (57)

$$f_0 = \left[\frac{\partial \Psi_0}{\partial T} - 2 \frac{\partial^2 \Psi_0}{\partial Y^2} - R_2 \iint^z D' \frac{\partial u_0}{\partial Y} \, dz^2 - R_0 \iint^z D' \frac{\partial u_2}{\partial Y} \, dz^2 - \iint^z \frac{\partial \Psi_0}{\partial Y} \frac{\partial^3 \Psi_0}{\partial z^3} \, dz^2 + \iint^z \frac{\partial \Psi_0}{\partial z} \frac{\partial^3 \Psi_0}{\partial z^2 \partial Y} \, dz^2\right]_{z=-1} + f_1$$

$$f_1 = \left[-\frac{\partial \Psi_0}{\partial T} + 2 \frac{\partial^2 \Psi_0}{\partial Y^2} + R_2 \iint^z D' \frac{\partial u_0}{\partial Y} \, dz^2 + R_0 \iint^z D' \frac{\partial u_2}{\partial Y} \, dz^2 + \iint^z \frac{\partial \Psi_0}{\partial Y} \frac{\partial^3 \Psi_0}{\partial z^3} \, dz^2 - \iint^z \frac{\partial \Psi_0}{\partial z} \frac{\partial^3 \Psi_0}{\partial z^2 \partial Y} \, dz^2\right]_{z=0}$$

$$f_2 = \left[-2 \iint^z \frac{\partial^2 \Psi_0}{\partial Y^2} \, dz^2 - R_2 \iiiint^z D' \frac{\partial u_0}{\partial Y} \, dz^4 - R_0 \iiiint^z D' \frac{\partial u_2}{\partial Y} \, dz^4 + \iint^z \frac{\partial \Psi_0}{\partial T} \, dz^2 - \iiiint^z \frac{\partial \Psi_0}{\partial Y} \frac{\partial^3 \Psi_0}{\partial z^3} \, dz^4 + \iiiint^z \frac{\partial \Psi_0}{\partial z} \frac{\partial^3 \Psi_0}{\partial z^2 \partial Y} \, dz^4\right]_{z=-1} - \frac{1}{6} f_0 + \frac{1}{2} f_1 + f_3$$

$$f_3 = \left[2 \iint^z \frac{\partial^2 \Psi_0}{\partial Y^2} \, dz^2 + R_2 \iiiint^z D' \frac{\partial u_0}{\partial Y} \, dz^4 + R_0 \iiiint^z D' \frac{\partial u_2}{\partial Y} \, dz^4 - \iint^z \frac{\partial \Psi_0}{\partial T} \, dz^2 + \iiiint^z \frac{\partial \Psi_0}{\partial Y} \frac{\partial^3 \Psi_0}{\partial z^3} \, dz^4 - \iiiint^z \frac{\partial \Psi_0}{\partial z} \frac{\partial^3 \Psi_0}{\partial z^2 \partial Y} \, dz^4\right]_{z=0}$$

---

## Appendix C

### A Theorem for a Class of Nonlinear Differential Equations

To extend the analysis to higher order and thus evaluate $R_{2j}$ for any $j > 0$, we created an algorithm which utilizes the following theorem, the essence of which is given in various texts (e.g. Muscalu and Schlag 2013):

**Theorem:** Provided that a $2L + 1$ term complex Fourier series of the exact general solution:

$$A = \sum_{n=-L}^{L} Q(A, e^{inlx}) e^{inlx} \quad (0 < l < \infty) \tag{C.1}$$

to an $M$th order ordinary differential equation:

$$\frac{d^M A}{d x^M} = \xi \tag{C.2}$$

exists, it determines the coefficients of $e^{inlx}$ for $n \in [-L, L]$ in the residual of (C.2) only if $\xi$, which must not contain $d^M A / dx^M$, is also expandable as a complex Fourier series as:

$$\xi = \sum_{n=-\infty}^{\infty} Q(\xi, e^{inlx}) e^{inlx} \quad (0 < l < \infty) \tag{C.3}$$

Herein $Q(f, e^{inlx})$ denotes the projection of $f$ onto $e^{inlx}$, while $A$ and $\xi$ are periodic with period $2\pi / l$ and all of their derivatives and integrals are continuous for all $x$.

**Proof:** Because the complex Fourier series of $A$ and $\xi$ exist, and because $A$ and $\xi$ are periodic with period $2\pi / l$ and all their derivatives and integrals are continuous for all $x$, we can integrate (C.2) $M$ times and substitute the result into (C.1) to find:

$$A = \sum_{n=-L}^{L} Q\left(\frac{d^{(-M)} \xi}{dx^{(-M)}}, e^{inlx}\right) e^{inlx} \tag{C.4}$$

where the notation $d^{(-M)} \xi / dx^{(-M)}$ denotes the $M$th integral of $\xi$ with respect to $x$. Substituting (C.4) into the residual $r$ of (C.2) then gives:

$$r = \frac{d^M}{dx^M} \sum_{n=-L}^{L} Q\left(\frac{d^{(-M)} \xi}{dx^{(-M)}}, e^{inlx}\right) e^{inlx} - \sum_{n=-\infty}^{\infty} Q\left(\xi, e^{inlx}\right) e^{inlx} \tag{C.5}$$

provided $\xi$ is expandable as a complex Fourier series as in (C.3). Equation (C.5) can then be written as:

$$r = -\sum_{n \notin [-L, L]} Q\left(\xi, e^{inlx}\right) e^{inlx} \tag{C.6}$$

which shows that the theorem is true.

---

## References

- Andrews, D.G. and McIntyre, M.E., An exact theory of nonlinear waves on a Lagrangian-mean flow. *J. Fluid Mech.* 1978, **89**, 609–646.
- Babanin, A.V., Ganopolski, A. and Phillips, W.R.C., Wave-induced upper-ocean mixing in a climate model of intermediate complexity. *Ocean Modelling* 2009, **29**, 189–197.
- Buhler, O., *Waves and Mean Flows*, 2009 (Cambridge University Press: Cambridge).
- Chapman, C.J. and Proctor, M.R.E., Nonlinear Rayleigh–Bénard convection between poorly conducting boundaries. *J. Fluid Mech.* 1980, **101**, 759–782.
- Chini, G.P., Strongly nonlinear Langmuir circulation and Rayleigh–Bénard convection. *J. Fluid Mech.* 2008, **614**, 39–65.
- Cox, S. and Leibovich, S., Langmuir circulations in a surface layer bounded by a strong thermocline. *J. Phys. Ocean.* 1993, **23**, 1330–1345.
- Craik, A.D.D., The generation of Langmuir circulations by an instability mechanism. *J. Fluid Mech.* 1977, **81**, 209–223.
- Craik, A.D.D., Wave induced longitudinal-vortex instability in shear flows. *J. Fluid Mech.* 1982, **125**, 37–52.
- Craik, A.D.D., *Waves Interactions and Fluid Flows*, 1985 (Cambridge University Press: Cambridge).
- Craik, A.D.D. and Leibovich, S., A rational model for Langmuir circulations. *J. Fluid Mech.* 1976, **73**, 401–426.
- Denman, K.L. and Gargett, A.E., Biological physical interactions in the upper ocean – The role of vertical and small-scale transport processes. *Annu. Rev. Fluid Mech.* 1995, **27**, 225–255.
- Gargett, A., Wells, J., Tejada-Martinez, A.E. and Grosch, C.E., Langmuir supercells: A mechanism for sediment resuspension and transport in shallow seas. *Science* 2004, **306**, 5703–5708.
- Gertsberg, V.L. and Sivashinsky, G.I., Large cells in nonlinear Rayleigh–Bénard convection. *Prog. Theor. Phys.* 1981, **66**, 1219–1229.
- Guasto, J.S., Rusconi, R. and Stocker, R., Fluid mechanics of planktonic microorganisms. *Annu. Rev. Fluid Mech.* 2012, **44**, 373–400.
- Hall, P. and Smith, F., The nonlinear interaction of Görtler vortices and Tollmien–Schlichting waves in curved channel flows. *Proc. R. Soc. Lond. A* 1988, **417**, 255–282.
- Hardy, G.H., *Divergent Series*, 1949 (Oxford University Press: Oxford).
- Hayes, D.T. and Phillips, W.R.C., An asymptotic study of instability to Langmuir circulation in shallow layers. *Geophys. Astrophys. Fluid Dyn.* 2016, **110**, 295–316.
- Holm, D.D., The ideal Craik–Leibovich equation. *Physica D* 1996, **98**, 415–441.
- Kenney, B.C., Observations of coherent bands of algae in a surface shear layer. *Limnol. Oceanogr.* 1993, **38**, 1059–1067.
- Langmuir, I., Surface motion of water induced by wind. *Science* 1938, **87**, 119–123.
- Leibovich, S., On wave–current interaction theories of Langmuir circulation. *J. Fluid Mech.* 1980, **99**, 715–724.
- Leibovich, S., The form and dynamics of Langmuir circulations. *Annu. Rev. Fluid Mech.* 1983, **15**, 391–427.
- Leibovich, S. and Paolucci, S., The instability of the ocean to Langmuir circulations. *J. Fluid Mech.* 1981, **102**, 141–167.
- Marmorino, G.O., Smith, G.B. and Lindemann, G.J., Infrared imagery of large-aspect-ratio Langmuir circulation. *Cont. Shelf Res.* 2005, **25**, 1–6.
- Melville, W.K., Shear, R. and Veron, F., Laboratory measurements of the generation and evolution of Langmuir circulations. *J. Fluid Mech.* 1998, **364**, 31–58.
- Muscalu, C. and Schlag, W., *Classical and Multilinear Harmonic Analysis*, Vol. 1, 2013 (Cambridge University Press: Cambridge).
- Newell, A.C., Passot, T. and Souli, M., The phase diffusion and mean drift equations for convection at finite Rayleigh numbers in large containers. *J. Fluid Mech.* 1990, **220**, 187–252.
- Nield, D.A., The thermohaline Rayleigh–Jeffreys problem. *J. Fluid Mech.* 1967, **29**, 545–558.
- Phillips, W.R.C., Finite-amplitude rotational waves in viscous shear flows. *Stud. Appl. Math.* 1998a, **101**, 23–47.
- Phillips, W.R.C., On the nonlinear instability of strong wavy shear to longitudinal vortices. In *Nonlinear Instability, Chaos and Turbulence*, edited by L. Debnath and D.N. Riahi, pp. 277–299, 1998b (WIT: Singapore).
- Phillips, W.R.C., On an instability to Langmuir circulations and the role of Prandtl and Richardson numbers. *J. Fluid Mech.* 2001a, **442**, 335–358.
- Phillips, W.R.C., On the pseudomomentum and generalized Stokes drift in a spectrum of rotational waves. *J. Fluid Mech.* 2001b, **430**, 209–229.
- Phillips, W.R.C., Langmuir circulations beneath growing or decaying surface waves. *J. Fluid Mech.* 2002, **469**, 317–342.
- Phillips, W.R.C., On the spacing of Langmuir circulation in strong shear. *J. Fluid Mech.* 2005, **525**, 215–236.
- Phillips, W.R.C., Drift and pseudomomentum in bounded turbulent shear flows. *Phys. Rev. E* 2015, **92**, 043003.
- Phillips, W.R.C. and Dai, A., On Langmuir circulation in shallow waters. *J. Fluid Mech.* 2014, **743**, 141–169.
- Phillips, W.R.C., Dai, A. and Tjan, K.K., On Lagrangian drift in shallow-water waves on moderate shear. *J. Fluid Mech.* 2010, **660**, 221–239.
- Phillips, W.R.C. and Shen, Q., A family of wave-mean shear interactions and their instability to longitudinal vortex form. *Stud. Appl. Math.* 1996, **96**, 143–161.
- Phillips, W.R.C. and Wu, Z., On the instability of wave-catalysed longitudinal vortices in strong shear. *J. Fluid Mech.* 1994, **272**, 235–254.
- Plueddemann, A., Smith, J., Farmer, D., Weller, R., Crawford, W., Pinkel, R., Vagle, S. and Gnanadesikan, A., Structure and variability of Langmuir circulation during the Surface Waves Processes Program. *J. Geophys. Res.* 1996, **21**, 85–102.
- Smith, J.A., Observed growth of Langmuir circulation. *J. Geophys. Res.* 1992, **97**, 5651–5664.
- Smith, J., Pinkel, R. and Weller, R.A., Velocity structure in the mixed layer during MILDEX. *J. Phys. Oceanogr.* 1987, **17**, 425–439.
- Stokes, G.G., On the theory of oscillatory waves. *Trans. Camb. Phil. Soc.* 1847, **8**, 441–455.
- Thorpe, S.A., Langmuir circulation. *Annu. Rev. Fluid Mech.* 2004, **36**, 55–79.
- Vladimirov, V.A., Proctor, M.R.E. and Hughes, D.W., Vortex dynamics of oscillatory flows. *Arnold Math. J.* 2015, **1**, 113–126.

---

## Notation Quick Reference for Implementers

This section summarises the notation conventions used throughout the paper for ease of implementation.

| Symbol | Meaning |
|---|---|
| $z$ | Vertical coordinate, $z \in [-1, 0]$, with $z=0$ at free surface, $z=-1$ at bottom |
| $y$ | Cross-stream (spanwise) coordinate |
| $x$ | Streamwise (wind-aligned) coordinate |
| $l$ | Spanwise wavenumber (the small expansion parameter) |
| $U(z)$ | Mean shear flow profile |
| $U'$ | $dU/dz$, the mean shear (polynomial in $z$, eq. 7b) |
| $D'$ | $dD/dz$, the differential drift (polynomial in $z$, eq. 7a) |
| $u$ | Streamwise perturbation velocity |
| $\psi$ | Streamfunction for cross-stream perturbation ($v = \psi_z$, $w = -\psi_y$) |
| $Ra$ | Rayleigh number $= U D h^2 / \nu_T^2$ |
| $R$ | Linear neutral curve value of $Ra$ as function of $l$ |
| $\bar{R}$ | Nonlinear neutral curve value of $Ra$ as function of $l$ |
| $R_0$ | Leading-order Rayleigh number (eq. 26) |
| $R_{2k}$ | Expansion coefficients: $R = \sum l^{2k} R_{2k}$ |
| $R^*_2$ | Neumann ($\gamma = 0$) part of nonlinear $R_2$ |
| $\tilde{R}_2$ | Robin correction part of nonlinear $R_2$ |
| $\sigma$ | Growth rate, expanded as $\sigma = \sum l^{2k+2} \sigma_{2k+2}$ |
| $\gamma_s$ | Robin BC parameter at free surface ($\approx 0.06$) |
| $\gamma_b$ | Robin BC parameter at bottom ($\approx 0.28$) |
| $\gamma$ | $\gamma_s + \gamma_b$ |
| $\tilde{\gamma}_s, \tilde{\gamma}_b$ | Rescaled: $\gamma_s = l^4 \tilde{\gamma}_s$, $\gamma_b = l^4 \tilde{\gamma}_b$ |
| $l_{c_L}$ | Critical linear wavenumber (eq. 67) |
| $l_{c_{NL}}$ | Critical nonlinear wavenumber (eq. 66) |
| $R_{c_L}$ | Critical linear Rayleigh number (eq. 70) |
| $R_{c_{NL}}$ | Critical nonlinear Rayleigh number (eq. 69) |
| $\kappa$ | Ratio $l_{c_L} / l_{c_{NL}}$ (eq. 68), independent of $\gamma$ |
| $\tilde{\psi}_1$ | Linear eigenfunction from eq. 24 (devoid of $R_0$) |
| $u_{2k}$ | $k$-th order expansion coefficient for $u$ |
| $\psi_{2k+1}$ | $k$-th order expansion coefficient for $\psi$ |
| $\hat{\psi}_{2j+1}$ | Part of $\psi_{2j+1}$ devoid of $R_{2j}$ |
| $\tilde{\psi}_{2j+1}$ | Part of $\psi_{2j+1}$ multiplied by $R_{2j}$ |
| $u_0(Y,T)$ | Leading-order nonlinear $u$, independent of $z$ (eq. 43) |
| $\Psi_0(Y,z,T)$ | Leading-order nonlinear streamfunction (eq. 46) |
| $Y$ | Rescaled spanwise coordinate $Y = ly$ |
| $T$ | Rescaled time $T = l^2 t$ |
| $J(a,b)$ | Jacobian: $a_y b_z - a_z b_y$ |
| $\nabla^2$ | Planar Laplacian: $\partial^2/\partial y^2 + \partial^2/\partial z^2$ |
| $P_m(z)$ | Shifted Legendre basis functions on $z \in [-1, 0]$ |
| $A_{m,k}(t)$ | Galerkin coefficients for $u$ (eq. 71) |
| $B_{m,k}(t)$ | Galerkin coefficients for $\psi$ (eq. 72) |
| $I$ | Number of Fourier harmonics retained (paper uses $I=1$) |
| $J$ | Number of Legendre modes (paper uses $J=13$) |
| $s$ | Shear index: $s=2$ open ocean, $s=1$ shallow coastal, $s=0$ lab |
| $\iint^z f \, dz^2$ | Double indefinite integral of $f$ with respect to $z$ |
| $\iiiint^z f \, dz^4$ | Quadruple indefinite integral of $f$ with respect to $z$ |
| $a_m$ | Polynomial coefficients for $D'$ (eq. 7a) |
| $b_n$ | Polynomial coefficients for $U'$ (eq. 7b) |
| $h_m$ | Fourier coefficients in $u_0$ expansion (eq. 52) |
| $\delta_{i,j}$ | Kronecker delta |

### Key Numerical Values from the Paper

| Quantity | Value | Context |
|---|---|---|
| $\gamma_s$ | $\approx 0.06$ | Surface Robin parameter (Cox & Leibovich 1993) |
| $\gamma_b$ | $\approx 0.28$ | Bottom Robin parameter (Cox & Leibovich 1993) |
| $R_{c_L}$ | $\approx 121.068$ | For $D' = U' = 1$, $\gamma_s = 0.0001$, $\gamma_b = 0$ |
| $l_{c_L}$ | $\approx 0.150$ | Same conditions |
| $R_{c_{NL}}$ | $\approx 122.194$ | Same conditions |
| $l_{c_{NL}}$ | $\approx 0.105$ | Same conditions |
| $\kappa$ for $(D',U') = (1,1)$ | $\approx 1.425$ | |
| $\kappa$ for $(D',U') = (1+z, 1)$ | $\approx 1.427$ | |
| $\kappa$ for $(D',U') = (1, 1+z)$ | $\approx 1.919$ | |
| $\kappa$ for $(D',U') = (1+z, 1+z)$ | $\approx 1.944$ | |
| Aspect ratio range | 5 to 11 | With $\gamma_s = 0.06$, $\gamma_b = 0.28$ |
| $J$ (Legendre modes) | 13 | Sufficient for all cases |
| $I$ (Fourier harmonics) | 1 | Used in numerical validation |
| Expansion order | $O(l^{16})$ | Asymptotic expansion taken to this order |
