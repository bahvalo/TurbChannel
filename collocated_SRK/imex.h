#ifndef IMEX_H
#define IMEX_H

#include <vector>

#define MAX_N_STAGES 6
#define BUTCHER_TABLE_MAX_SIZE (MAX_N_STAGES*MAX_N_STAGES+MAX_N_STAGES)

enum struct tIMEXtype{ IMEX_A, IMEX_ARS, IMEX_CK };

// Abstract class for an IMEX method (the double Butcher table and related information)
struct tIMEXMethod{
    // Type of the IMEX methods. A1 means that the method is of type A but uses the same IMEX formulation as methods of other types
    tIMEXtype IMEXtype=tIMEXtype::IMEX_A;
    int NStages = 0; // number of stages of the RK method
    std::vector<double> ButcherE, ButcherI; // Butcher tables for the explicit and implicit methods (of size (NStages+1)*NStages)
    double alpha_tau_max[2] = {2.,2.}; // Stability limit in alpha_tau for StabType=0 and StabType=1

    static std::vector<double> CalcD(int NStages, const std::vector<double>& A); // Evaluate coefficients `d' for a Butcher table
    static void OrderCheck(int NumStages, const std::vector<double>& A); // Check of some order conditions for a Butcher table
protected:
    tIMEXMethod() {} // objects of the class tIMEXMethod are not allowed. Objects of derived classes only
};


// Specific IMEX methods of type ARS
// Ascher, Ruuth, Spiteri. IMEX RK method for time-dependent PDEs. 1997
struct ARS_121: tIMEXMethod{ ARS_121(); };
struct ARS_232: tIMEXMethod{ ARS_232(); };
struct ARS_343: tIMEXMethod{ ARS_343(); };
// Boscarino, Russo. On the uniform accuracy of IMEX RK schemes and applications to hyperbolic systems with relaxation. 2007
struct MARS_343: tIMEXMethod{ MARS_343(); }; // MARS (=modified ARS) (3,4,3) method

// Specific IMEX methods of type CK
// Kennedy, Carpenter. Additive Runge-Kutta schemes for convection-diffusion-reaction equations. 2003
struct ARK3 : tIMEXMethod{ ARK3(); }; // ARK3(2)4L[2]SA scheme
struct ARK4 : tIMEXMethod{ ARK4(); }; // ARK4(3)6L[2]SA scheme
// Boscarino, Russo. On the uniform accuracy of IMEX RK schemes and applications to hyperbolic systems with relaxation. 2007
struct MARK3: tIMEXMethod{ MARK3(); }; // MARK3(2)4L[2]SA scheme
// Boscarino. On an accurate third order IMEX RK method for stiff problems. 2009
struct BHR_553: tIMEXMethod{ BHR_553(); };

#endif
