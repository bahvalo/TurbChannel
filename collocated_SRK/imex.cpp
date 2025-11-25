#include <stdio.h>
#include "imex.h"
using namespace std;


// Check some order conditions for a Butcher table
void tIMEXMethod::OrderCheck(int NumStages, const vector<double>& A){
    if(int(A.size()) != NumStages*(NumStages+1)) { printf("OrderCheck: wrong input\n"); exit(0); }
    const double* b = A.data() + NumStages*NumStages;
    double sum1 = 0., sum2 = 0., sum31 = 0., sum32 = 0.;

    // First order condition
    for(int k=0; k<NumStages; k++) sum1 += b[k];
    sum1 -= 1.0;

    // Second order condition
    for(int j=0; j<NumStages; j++) for(int k=0; k<NumStages; k++) sum2 += 2.*b[j]*A[j*NumStages+k];
    sum2 -= 1.0;

    // Third order conditions (H.,N.,W., Section II.2)
    for(int j=0; j<NumStages; j++) for(int k=0; k<NumStages; k++) for(int l=0; l<NumStages; l++)
        sum31 += 3.*b[j]*A[j*NumStages+k]*A[j*NumStages+l];
    sum31 -= 1.0;

    for(int j=0; j<NumStages; j++) for(int k=0; k<NumStages; k++) for(int l=0; l<NumStages; l++)
        sum32 += 6.*b[j]*A[j*NumStages+k]*A[k*NumStages+l];
    sum32 -= 1.0;

    printf("sums: %.2e, %.2e, %.2e, %.2e\n", sum1, sum2, sum31, sum32);
}


// Calculate coefficients `d`, which are used for methods of type CK
vector<double> tIMEXMethod::CalcD(int NumStages, const vector<double>& A){
    vector<double> d(NumStages);
    double a_ss = A[NumStages*NumStages-1];
    d[0] = 1.;
    for(int j=1; j<NumStages; j++){
        d[j] = 0.;
        for(int k=0; k<j; k++) d[j] -= A[j*NumStages+k]*d[k]/a_ss;
    }
    return d;
}


// Ascher, Ruuth, Spiteri. IMEX RK method for time-dependent PDEs. 1997
// Implicit-explicit Euler method
ARS_121::ARS_121(){
    NStages = 2;
    IMEXtype = tIMEXtype::IMEX_ARS;
    ButcherE = vector<double>({
        0., 0.,
        1., 0.,
        0., 1.
    });
    ButcherI = vector<double>({
        0., 0.,
        0., 1.,
        0., 1.
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// ARS (2,3,2) method
ARS_232::ARS_232(){
    NStages = 3;
    IMEXtype = tIMEXtype::IMEX_ARS;
    const double gamma = 1.-0.5*sqrt(2.);
    const double delta = -2.*sqrt(2.)/3.;
    ButcherE = vector<double>({
           0.,       0.,    0.,
        gamma,       0.,    0.,
        delta, 1.-delta,    0.,
           0., 1.-gamma, gamma
    });
    ButcherI = vector<double>({
           0.,       0.,    0.,
           0.,    gamma,    0.,
           0., 1.-gamma, gamma,
           0., 1.-gamma, gamma
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// ARS (3,4,3) method
ARS_343::ARS_343(){
    NStages = 4;
    IMEXtype = tIMEXtype::IMEX_ARS;
    const double gamma = 0.435866521508458999416019451193556842529; // middle root of 6x^3-18x^2+9x-1=0
    const double b1 = -1.5*gamma*gamma + 4*gamma - 0.25;
    const double b2 = 1.5*gamma*gamma - 5.*gamma + 1.25;
    ButcherI = vector<double>({
           0.,            0.,    0.,    0.,
           0.,         gamma,    0.,    0.,
           0., (1.-gamma)/2., gamma,    0.,
           0.,            b1,    b2, gamma,
           0.,            b1,    b2, gamma
    });
    const double a42 = 0.5529291479;
    const double a43 = a42;
    const double a31 = (1.-4.5*gamma+1.5*gamma*gamma)*a42 + (2.75-10.5*gamma+3.75*gamma*gamma)*a43 - 3.5+13.*gamma-4.5*gamma*gamma;
    const double a32 = -(1.-4.5*gamma+1.5*gamma*gamma)*a42 - (2.75-10.5*gamma+3.75*gamma*gamma)*a43 + 4.-12.5*gamma+4.5*gamma*gamma;
    ButcherE = vector<double>({
              0.,        0.,    0.,    0.,
           gamma,        0.,    0.,    0.,
             a31,       a32,    0.,    0.,
      1.-a42-a43,       a42,   a43,    0.,
              0.,        b1,    b2, gamma
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// Boscarino, Russo. On the uniform accuracy of IMEX RK schemes and applications to hyperbolic systems with relaxation. 2007
// MARS (=modified ARS) (3,4,3) method
MARS_343::MARS_343(){
    NStages = 4;
    IMEXtype = tIMEXtype::IMEX_ARS;
    const double gamma = 0.435866521508458999416019451193556842529; // middle root of 6x^3-18x^2+9x-1=0
    const double b2 = 1.20849664917601;
    ButcherI = vector<double>({
        0.0,                0.0,               0.0,            0.0,
        0.0,              gamma,               0.0,            0.0,
        0.0,   0.28206673924577,             gamma,            0.0,
        0.0,                 b2,       1.-gamma-b2,          gamma,
        0.0,                 b2,       1.-gamma-b2,          gamma
    });
    ButcherE = vector<double>({
                     0.0,                0.0,               0.0,            0.0,
                   gamma,                0.0,               0.0,            0.0,
        0.535396540307354, 0.182536720446875,               0.0,            0.0,
        0.63041255815287, -0.83193390106308,   1.20152134291021,            0.0,
        0.0,                 b2,       1.-gamma-b2,          gamma
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// ARK3(2)4L[2]SA scheme
ARK3::ARK3(){
    NStages = 4;
    IMEXtype = tIMEXtype::IMEX_CK;
    const double gamma = 0.435866521508458999416019451193556842529; // middle root of 6x^3-18x^2+9x-1=0
    const double b1 = 1471266399579./7840856788654.;
    const double b2 = -4482444167858./7529755066697.;
    const double b3 = 1.-gamma-b1-b2;
    ButcherI = vector<double>({
                      0.0,                0.0,               0.0,            0.0,
                    gamma,              gamma,               0.0,            0.0,
        2746238789719./10658868560708.,  -640167445237./6845629431997.,  gamma, 0.0,
        b1, b2, b3, gamma,
        b1, b2, b3, gamma
    });
    ButcherE = vector<double>({
                      0.0,                0.0,               0.0,            0.0,
                  2*gamma,                0.0,               0.0,            0.0,
        5535828885825./10492691773637., 788022342437./10882634858940., 0.0, 0.0,
        6485989280629./16251701735622., -4246266847089./9704473918619., 10755448449292./10357097424841., 0.0,
        b1, b2, b3, gamma
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// ARK4(3)6L[2]SA scheme
ARK4::ARK4(){
    NStages = 6;
    IMEXtype = tIMEXtype::IMEX_CK;
    const double b[6] = {82889./524892., 0., 15625./83664., 69875./102672., -2260./8211., 0.25};
    ButcherI = vector<double>({
                      0.0,                0.0,               0.0,            0.0,            0.0,            0.0,
                     0.25,               0.25,               0.0,            0.0,            0.0,            0.0,
             8611./62500.,      -1743./31250.,              0.25,            0.0,            0.0,            0.0,
       5012029./34652500.,  -654441./2922500.,   174375./388108.,           0.25,            0.0,            0.0,
       15267082809./155376265600., -71443401./120774400., 730878875./902184768., 2285395./8070912., 0.25,    0.0,
                     b[0],               b[1],               b[2],           b[3],          b[4],           b[5],
                     b[0],               b[1],               b[2],           b[3],          b[4],           b[5]
    });
    ButcherE = vector<double>({
                      0.0,                0.0,               0.0,            0.0,            0.0,            0.0,
                      0.5,                0.0,               0.0,            0.0,            0.0,            0.0,
            13861./62500.,       6889./62500.,               0.0,            0.0,            0.0,            0.0,
            -116923316275./2393684061468., -2731218467317./15368042101831., 9408046702089./11113171139209., 0.0, 0.0, 0.0,
            -451086348788./2902428689909., -2682348792572./7519795681897., 12662868775082./11960479115383., 3355817975965./11060851509271., 0.0, 0.0,
            647845179188./3216320057751., 73281519250./8382639484533., 552539513391./3454668386233., 3354512671639./8306763924573., 4040./17871., 0.0,
                     b[0],               b[1],               b[2],           b[3],          b[4],           b[5]
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// MARK3(2)4L[2]SA scheme
MARK3::MARK3(){
    NStages = 4;
    IMEXtype = tIMEXtype::IMEX_CK;
    const double gamma = 0.435866521508458999416019451193556842529; // middle root of 6x^3-18x^2+9x-1=0
    const double b1 = 0.60424832458800;
    ButcherI = vector<double>({
                      0.0,                0.0,               0.0,            0.0,
                    gamma,              gamma,               0.0,            0.0,
        -4.30002662176923,   2.26541338346372,             gamma,            0.0,
                       b1,                0.0,       1.-gamma-b1,          gamma,
                       b1,                0.0,       1.-gamma-b1,          gamma
    });
    ButcherE = vector<double>({
                      0.0,                0.0,               0.0,            0.0,
                  2*gamma,                0.0,               0.0,            0.0,
        -3.06478674186224,   1.46604002506519,               0.0,            0.0,
         0.21444560762133,   0.71075364965269,  0.07480074272597,            0.0,
                       b1,                0.0,       1.-gamma-b1,          gamma
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// Boscarino. On an accurate third order IMEX RK method for stiff problems. 2009
// Boscarino (5,5,3) method
BHR_553::BHR_553(){
    NStages = 5;
    ButcherI.resize(30);
    ButcherE.resize(30);
    IMEXtype = tIMEXtype::IMEX_CK;
    const double gamma = 424782./974569.;
    ButcherI[5]=ButcherI[6]=gamma;
    ButcherI[10]=gamma; ButcherI[11]=-31733082319927313./455705377221960889379854647102.; ButcherI[12]=gamma;
    ButcherI[15]=-3012378541084922027361996761794919360516301377809610./45123394056585269977907753045030512597955897345819349.;
    ButcherI[16]=-62865589297807153294268./102559673441610672305587327019095047.;
    ButcherI[17]=418769796920855299603146267001414900945214277000./212454360385257708555954598099874818603217167139.;
    ButcherI[18]=gamma;
    ButcherI[20]=487698502336740678603511./1181159636928185920260208.;
    ButcherI[22]=302987763081184622639300143137943089./1535359944203293318639180129368156500.;
    ButcherI[23]=-105235928335100616072938218863./2282554452064661756575727198000.;
    ButcherI[24]=gamma;
    ButcherE[5]=2.*gamma;
    ButcherE[10]=gamma; ButcherE[11]=gamma;
    ButcherE[15]=-475883375220285986033264./594112726933437845704163.;
    ButcherE[17]=1866233449822026827708736./594112726933437845704163.;
    ButcherE[20]=62828845818073169585635881686091391737610308247./176112910684412105319781630311686343715753056000.;
    ButcherE[21]=-ButcherI[22];
    ButcherE[22]=262315887293043739337088563996093207./297427554730376353252081786906492000.;
    ButcherE[23]=-987618231894176581438124717087./23877337660202969319526901856000.;
    for(int i=0; i<5; i++) ButcherE[25+i]=ButcherI[25+i]=ButcherI[20+i];
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}
