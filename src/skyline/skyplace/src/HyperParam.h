#ifndef HYPERPARAM_H
#define HYPERPARAM_H

#include <cstdio>

namespace skyplace
{

class HyperParam
{
  public:
    HyperParam()
    {
      maxOptIter        =      5000;
      maxBackTrackIter  =        10;
      initLambda        =      0.0007; // 0.01
      initGammaInvCoef  =      0.0125; // 0.0125
      targetOverflow    =      0.10;
      minPhiCoef        =      0.95;
      maxPhiCoef        =      1.05; 
      minPrecond        =       1.0;
      initOptCoef       =       100;
      referenceHpwl     =    350000;
      //referenceHpwl     = 446000000;
      minStepLength     =       1.0;
      adam_alpha        =     100.0;
      adam_beta1        =      0.90;
      adam_beta2        =     0.999;
    }

    int maxOptIter;
    int maxBackTrackIter;

    float initLambda;
    float initGammaInvCoef;
    float targetOverflow;
    float minPhiCoef;
    float maxPhiCoef;
    float minPrecond;
    float initOptCoef;
    float referenceHpwl;
    float minStepLength;

    // Adam-related
    float adam_alpha;
    float adam_beta1;
    float adam_beta2;

    void printHyperParameters()
    {
      printf("\n");
      printf("  -----------------------------------\n");
      printf("  |         Hyper Parameters        |\n");
      printf("  -----------------------------------\n");
      printf("  | maxOptIter     | %12d   |\n", maxOptIter);
      printf("  | maxBackTrack   | %12d   |\n", maxBackTrackIter);
      printf("  | initLambda     | %12.6f   |\n", initLambda);
      printf("  | initGammaInvCo | %12.6f   |\n", initGammaInvCoef);
      printf("  | TargetOverflow | %12.6f   |\n", targetOverflow);
      printf("  | minPhiCoef     | %12.6f   |\n", minPhiCoef);
      printf("  | maxPhiCoef     | %12.6f   |\n", maxPhiCoef);
      printf("  | minPrecond     | %12.1f   |\n", minPrecond);
      printf("  | initOptCoef    | %12.1f   |\n", initOptCoef);
      printf("  | referenceHpwl  | %12.1f   |\n", referenceHpwl);
      printf("  | minStepLength  | %12.1f   |\n", minStepLength);
      printf("  -----------------------------------\n");
    }
};

}; // namespace skyplace

#endif 
