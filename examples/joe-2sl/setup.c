#include <stdio.h>
#include "refl.h"
#include "reflcalc.h"

typedef double (*Alloy)(double, double, double, double, double, double, double);
double volume_fraction(double p, double m1, double d1, double r1, 
                                 double m2, double d2, double r2)
{
  double vf = p;
  //printf("vf=%g r1=%g r2=%g\n",vf,r1,r2);
  return vf*r1 + (1-vf)*r2;
}
double mass_fraction(double p, double m1, double d1, double r1, 
                               double m2, double d2, double r2)
{
  double v1 = p/d1, v2 = (1-p)/d2;
  double vf = v1/(v1+v2);
  return vf*r1 + (1-vf)*r2;
}
double atomic_fraction(double p, double m1, double d1, double r1, 
                                 double m2, double d2, double r2)
{
  double v1 = p*m1/d1, v2=(1-p)*m2/d2;
  double vf = v1/(v1+v2);
  return vf*r1 + (1-vf)*r2;
}



#define MODELS 1

int stack1_size = 14;
int stack2_size = 100-14;

/* Physical properties */
double MgO_density =  3.58;
double   V_density =  6.0;
double  Fe_density =  7.87;
double  Pd_density = 12.023;
double   V_mass = 50.9415;
double  Fe_mass = 55.845;

/* Scattering length densities */
double MgO_rho =  5.98031e-6, MgO_mu = 0.04701e-9;
double   V_rho = -0.31422e-6,   V_mu = 5.01208e-9;
double  Fe_rho =  8.02000e-6,  Fe_mu = 3.02208e-9;
double  Pd_rho =  4.02094e-6,  Pd_mu = 6.53001e-9;

/* Density tweaks: scaling from natural density */
double MgO_packing = 1.0;
double   V_packing = 1.0;
double FeV_packing = 1.0;
double  Pd_packing = 1.0;

/* Alloy parameters: portion of Fe in FeV layers */
Alloy  FeV_alloy = mass_fraction;
double Fe_fraction = 0.25;

/* Structure */
double Pd_thickness = 100;
double stack1_V_thickness = 26;
double stack1_FeV_thickness = 18.1;
double stack2_V_thickness = 26;
double stack2_FeV_thickness = 11;
double MgO_V_roughness = 6/2.35;
double V_FeV_roughness = 6/2.35;
double V_Pd_roughness = 30/2.35;
double Pd_Air_roughness = 30/2.35;

/*=========== CONSTRAINTS =====================*/
void constr_models(fitinfo *fit)
{
  int i,k,n;

  /* Set properties for all layers in all models */
  for (i=0; i< MODELS; i++) {
    n = 1;

    /* Pd cap */
    fit[i].m.rho[n] = Pd_packing * Pd_rho;
    fit[i].m.mu[n] = Pd_packing * Pd_mu;
    fit[i].m.d[n] = Pd_thickness;
    fit[i].m.rough[n] = Pd_Air_roughness;
    n++;

    /* Final V */
    fit[i].m.rho[n] = V_packing * V_rho;
    fit[i].m.mu[n] = V_packing * V_mu;
    fit[i].m.d[n] = stack2_V_thickness;
    fit[i].m.rough[n] = V_Pd_roughness;
    n++;

    /* Stack 1 V/FeV */    
    for (k=0; k < stack1_size; k++) {
      fit[i].m.rho[n] = V_packing * V_rho;
      fit[i].m.mu[n] = V_packing * V_mu;
      fit[i].m.d[n] = stack1_V_thickness;
      fit[i].m.rough[n] = V_FeV_roughness;
      n++;
      fit[i].m.rho[n] = FeV_packing*(*FeV_alloy)(Fe_fraction,
                        Fe_mass, Fe_density, Fe_rho, 
                         V_mass,  V_density,  V_rho);
      fit[i].m.mu[n]  = FeV_packing*(*FeV_alloy)(Fe_fraction,
                        Fe_mass, Fe_density, Fe_mu, 
                         V_mass,  V_density,  V_mu);
      fit[i].m.d[n]   = stack1_FeV_thickness;
      fit[i].m.rough[n] = V_FeV_roughness;
      n++;
    }

    /* Stack 2 V/FeV */    
    for (k=0; k < stack2_size; k++) {
      fit[i].m.rho[n] = V_packing * V_rho;
      fit[i].m.mu[n] = V_packing * V_mu;
      fit[i].m.d[n] = stack2_V_thickness;
      fit[i].m.rough[n] = V_FeV_roughness;
      n++;
      fit[i].m.rho[n] = FeV_packing*(*FeV_alloy)(Fe_fraction,
                        Fe_mass, Fe_density, Fe_rho, 
                         V_mass,  V_density,  V_rho);
      fit[i].m.mu[n]  = FeV_packing*(*FeV_alloy)(Fe_fraction,
                        Fe_mass, Fe_density, Fe_mu, 
                         V_mass,  V_density,  V_mu);
      fit[i].m.d[n]   = stack2_FeV_thickness;
      fit[i].m.rough[n] = V_FeV_roughness;
      n++;
    }

    /* MgO */
    fit[i].m.rho[n] = MgO_packing * MgO_rho;
    fit[i].m.mu[n] = MgO_packing * MgO_mu;
    fit[i].m.d[n] = 0; /* Semi-infinite substrate */
    fit[i].m.rough[n] = MgO_V_roughness;
    n++;
 
  }
}



fitinfo* setup_models(int *models)
{
  static fitinfo fit[MODELS];
  fitpars *pars = &fit[0].pars;
  int i, k;

  *models = MODELS;

  for (i=0; i < MODELS; i++) fit_init(&fit[i]);

  /* Load the data for each model */
  fit_data(&fit[0],"12v2b006_nobkgr_corr.refl");

  /* Initialize instrument parameters for each model.*/
  for (i=0; i < MODELS; i++) {
    const double L = 5.0042,dLoL=0.020,d=1890.0;
    double Qlo,Tlo, dTlo,dToT,s1,s2;
    Qlo=0.0154,Tlo=0.35;
    s1=0.21, s2=s1;
    dTlo=resolution_dT(s1,s2,d);
    dToT=resolution_dToT(s1,s2,d,Tlo);
    data_resolution_fv(&fit[i].dataA,L,dLoL,Qlo,dTlo,dToT);
    fit[i].beam.lambda = L;
    fit[i].beam.background = 1e-10;
    interface_create(&fit[i].rm, "erf", erf_interface, 30);
  }

  /*============= MODEL =====================================*/

  /* Add layers: d (thickness), rho (Nb), mu (absorption), rough (interface) */
  for (i=0; i < MODELS; i++) {
    /* Allocate space for 2*#stacks plus MgO substrate, additional V,
       Pd cap and Air */
    for (k=0; k < 2*(stack1_size+stack2_size)+4; k++) {
        model_layer(&fit[i].m, 0, 0, 0, 0); 
    } 
  }

  /* Structure */
  pars_add(pars, "Fe fraction", &Fe_fraction, 0.1, 1);
  pars_add(pars, "d V1", &stack1_V_thickness, 1, 50);
  pars_add(pars, "d FeV1", &stack1_FeV_thickness, 1, 50);
  pars_add(pars, "d V2", &stack2_V_thickness, 1, 50);
  pars_add(pars, "d FeV2", &stack2_FeV_thickness, 1, 50);
  pars_add(pars, "d Pd", &Pd_thickness, 1, 200);
  pars_add(pars, "MgO:V", &MgO_V_roughness, 0, 10);
  pars_add(pars, "V:FeV", &V_FeV_roughness, 0, 10);
  pars_add(pars, "V:Pd", &V_Pd_roughness, 0, 10);
  pars_add(pars, "Pd:Air", &Pd_Air_roughness, 0, 10);
  pars_add(pars, "FeV packing", &FeV_packing, 0.9, 1.1);
  pars_add(pars, "V packing", &V_packing, 0.9, 1.1);
  pars_add(pars, "Pd packing", &Pd_packing, 0.9, 1.1);
  pars_add(pars, "MgO packing", &MgO_packing, 0.9, 1.1);

  constraints = constr_models;
  return fit;
}
