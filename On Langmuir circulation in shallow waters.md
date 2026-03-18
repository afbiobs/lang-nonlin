# **On Langmuir circulation in shallow waters**

**W. R. C. Phillips and A. Dai** *J. Fluid Mech. (2014), vol. 743, pp. 141–169.*

## **Abstract**

The instability of shallow-water waves on a moderate shear to Langmuir circulation is considered. In such instances, specifically at the shallow end of the inner coastal region, the shear can significantly affect the drift giving rise to profiles markedly different from the simple Stokes drift. Since drift and shear are instrumental in the instability to Langmuir circulation, of key interest is how that variation in turn affects onset to Langmuir circulation. Also of interest is the effect on onset of various boundary conditions. To that end the initial value problem describing the wave-mean flow interaction which accounts for the multiple time scales of the surface waves, evolving shear and evolving Langmuir circulation is crafted from scratch, and includes the wave-induced drift and a consistent set of free-surface boundary conditions.

## **1\. Introduction**

It has long been known that the Stokes drift in irrotational water waves in otherwise quiescent surroundings transitions from an exponential decay with depth in deep-water waves (Stokes 1847\) to quadratic decay in shallow-water waves (Longuet-Higgins 1953). However, that need not be the case when the waves travel on a shear layer of sufficient strength.  
Phillips et al. (2010) restrict attention to small-amplitude waves in which the drift $d$ takes the form:  
$$d\_{i} \= \\overline{\\xi\_{j}\\tilde{u}\_{i,j}} \+ \\frac{1}{2}\\overline{\\xi\_{j}\\xi\_{k}}\\bar{u}\_{i,jk} \+ O(\\epsilon^{3})$$  
(1.1)  
where indices $(1,2,3) \\mapsto (x,y,z)$ with unit vectors $(\\mathbf{i}, \\mathbf{j}, \\mathbf{k})$, repeated indices imply summation and commas denote partial differentiation. $\\bar{u}$ is the mean Eulerian velocity, $\\tilde{u}$ is the Eulerian fluctuating velocity and $\\xi$ is the associated particle displacement field.  
For a unidirectional shear flow:

$$\\bar{u} \= \\epsilon^{s}\[U(z,t), 0, 0\]$$  
(1.2)  
The physical requirement $a/h \\ll 1$ is satisfied only when $s=0$ or $s=1$. Consequently, the second term in (1.1) must be retained whenever $s \\in \[0,1\]$.

## **2\. The CLg equations**

### **2.1. GLM theory**

The Generalized Lagrangian-mean (GLM) equations for homentropic flows of constant density $\\rho$ in a non-rotating reference frame are:  
$$\\bar{q}\_{i,t} \+ \\bar{q}\_{j}\\bar{q}\_{i,j} \- p\_{j}(\\bar{q}\_{j,i} \- \\bar{q}\_{i,j}) \+ \\Pi\_{i} \= \\mathcal{X}\_{i}$$  
(2.2)  
where $\\bar{q} \= \\bar{u}^{L} \- p$ and $\\mathcal{X}$ allows for dissipative forces.

### **2.2. Imposed shear and waves**

The velocity field $u(x,y,z,t)$ is decomposed as:

$$u \= \\bar{U} \+ \\tilde{u} \+ u' \= \\epsilon^{s}\\{U(z,t) \+ \\Delta u(y,z,t)\\} \+ \\epsilon\\tilde{U}(x,z,t) \+ O(\\epsilon^{s+1}\\Delta)$$  
(2.4)  
The pseudomomentum $p$ is expanded as:

$$p(y,z,t) \= \\epsilon^{2}\\{\[P\_{1}, 0, P\_{3}\] \+ \\epsilon^{s}\\Delta\[p\_{1}, \\epsilon^{n}p\_{2}, \\epsilon^{n}p\_{3} \+ \\dots\]\\}$$  
(2.6)

### **2.3. The secondary flow field**

For structure arising through an instability requiring $\\partial P\_{1}/\\partial z \\neq 0$, the evolution equations for $q\_1$ and the vorticity field $\\mathcal{U}\_1$ are:  
$$\\frac{\\partial q\_{1}}{\\partial t} \+ \\Delta\\left(q\_{2}\\frac{\\partial q\_{1}}{\\partial y} \+ q\_{3}\\frac{\\partial q\_{1}}{\\partial z}\\right) \+ \\epsilon^{(2-s)/2}D\_{3}\\frac{\\partial q\_{1}}{\\partial z} \+ q\_{3}\\frac{\\partial Q\_{1}}{\\partial z} \= \\epsilon^{-(s+2)/2}\\mathfrak{R}^{-1}\\nabla^{2}q\_{1} \+ O(\\epsilon^{(2-s)/2}\\mathfrak{R}^{-1})$$  
(2.11)  
$$\\frac{\\partial\\mathcal{U}\_{1}}{\\partial t} \+ \\Delta\\left(\\frac{\\partial\\mathcal{U}\_{1}q\_{2}}{\\partial y} \+ \\frac{\\partial\\mathcal{U}\_{1}q\_{3}}{\\partial z}\\right) \+ \\epsilon^{(2-s)/2}\\frac{\\partial}{\\partial z}(\\mathcal{U}\_{1}D\_{3}) \+ \\frac{\\partial q\_{1}}{\\partial y}\\frac{\\partial P\_{1}}{\\partial z} \- \\epsilon^{s}\\frac{\\partial Q\_{1}}{\\partial z}\\frac{\\partial p\_{1}}{\\partial y} \= \\dots$$  
(2.12)

## **3\. Wave field and boundary conditions**

### **3.1. Wave equations**

For monochromatic two-dimensional waves:

$$\\tilde{u}(x,y,z,t) \= \\epsilon\[\\Phi', 0, \-i\\alpha\\Phi\]e^{i\\beta} \+ \\epsilon^{s+1}\[\\phi', 0, \-i\\alpha\\phi\]e^{i\\beta} \+ \\tilde{u}^{\*}$$  
(3.1)  
The $O(\\epsilon)$ irrotational waves satisfy:

$$\\Phi'' \- \\alpha^{2}\\Phi \= 0$$  
(3.3)  
With the solution:

$$\\Phi \= \\frac{\\omega}{\\alpha^{2}}\\frac{\\sinh \\alpha(z+1)}{\\sinh \\alpha} \\quad \\text{with} \\quad \\omega \= \\sqrt{g\\alpha \\tanh \\alpha}$$  
(3.5a,b)

### **3.2. Free-surface boundary conditions**

At $z=0$, the consistent set of boundary conditions for $s=1$ are derived as Cauchy conditions:  
For the axial component $u$:

$$u' \- \\frac{(\\alpha^{2} \+ l^{2})\\Phi'}{\\alpha^{2}\\Phi}u \= 0$$  
(3.12a)  
For the vertical component $w$:

$$w' \+ \\frac{\\Phi' w}{\\Phi} \= 0$$  
(3.13a)  
At the rigid bottom $z \= \-1$:

$$U \= 0, \\quad u \= 0, \\quad w' \= w \= 0$$  
(3.14)

## **4\. Initial value problem**

The evolution equations (4.2) are solved numerically:  
$$\\frac{\\partial U}{\\partial \\tau} \- \\frac{\\partial^{2}U}{\\partial z^{2}} \= \-G$$  
(4.2a)

$$\\frac{\\partial u}{\\partial t} \- (D^{2} \- l^{2})u \= \-v U'$$  
(4.2b)

$$(D^{2} \- l^{2})\\frac{\\partial w}{\\partial t} \- (D^{2} \- l^{2})^{2}w \= \\mathcal{R}l^{2}u D'$$  
(4.2c)  
Where the Rayleigh number is defined as:

$$\\mathcal{R} \= \\frac{\\mathbb{H}\\mathbb{G}h^{2}}{\\nu\_{T}^{2}}$$  
(4.3)

## **5\. Drift**

The differential drift $D'$ for $s=1$ takes the form:  
$$D' \= \\frac{1}{2}\\text{csch}^{2}\\alpha \\{2\\alpha \\sinh 2\\alpha(z+1) \+ \\vartheta\[2\\alpha U'' \\sinh 2\\alpha(z+1) \- \\frac{1}{2}\[U'' \- (U'' \+ 4\\alpha^{2}U')\\cosh 2\\alpha(z+1)\]\]\\}$$  
(4.9)  
with:

$$\\vartheta \= \\frac{\\epsilon}{2\\alpha^{3/2}\\sqrt{g \\tanh \\alpha}}$$  
(4.10)  
In the limit $\\alpha \\rightarrow 0$ with $U' \= 0$:

$$D' \\sim 2(z+1)$$  
(4.11)

## **7\. Discussion**

The study finds that Langmuir circulation excited by the CL2 instability can drive contiguous, less intense circulations in a stacked array. Omission of the shear-related component of drift (using only Stokes drift) can yield misleading results in pressure-driven flows in shallow water.