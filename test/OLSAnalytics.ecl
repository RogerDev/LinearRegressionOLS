/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
IMPORT $.^ as LROLS;
IMPORT ML_Core;
IMPORT ML_Core.Types as mlTypes;
IMPORT PBblas;
IMPORT PBBlas.test.MakeTestMatrix as tm;
IMPORT PBBlas.Types as pbbTypes;
IMPORT PBBlas.Converted as pbbConverted;
IMPORT ML_Core.Math as Math;
Layout_Cell := pbbTypes.Layout_Cell;
NumericField := mlTypes.NumericField;
two31 := POWER(2, 31);  // 2**31
epsilon := .01;  // Allowable error
REAL Noise(maxv=.1) := FUNCTION
  out := ((RANDOM()-two31)%1000000)/(1000000/maxv);
  return out;
END;
// Make two sets of test Xs and Ys so that we can test the analytics
// using Myriad.  Let set 2 be many times as noisy as set 1.

// Set 1 is a simple linear regression (i.e. single independent variable)

A1 := 3.123;
B1 := -1.222;
N1 := 1000;
M1 := 1;
mX1 := tm.Random(N1, M1, 1.0, 1);
Layout_Cell makeY1(Layout_Cell X, REAL A, REAL B) := TRANSFORM
  SELF.x := X.x;
  SELF.y := 1;
  SELF.wi_id := X.wi_id;
  SELF.v := A1 * X.v  + B1 + Noise(10);
END;
mY1 := PROJECT(mX1, makeY1(LEFT, A1, B1));
X1 := pbbConverted.MatrixToNF(mX1);
Y1 := pbbConverted.MatrixToNF(mY1);

// Set 2 is a multiple linear regression with 3 independents
compX2 := RECORD
  REAL wi;
  REAL id;
  REAL X1;
  REAL X2;
  REAL X3;
END;
compX2 makeComposite2(Layout_Cell l, DATASET(Layout_Cell) r) := TRANSFORM
  SELF.wi := l.wi_id;
  SELF.id := l.x;
  SELF.X1 := r(y=1)[1].v;
  SELF.X2 := r(y=2)[1].v;
  SELF.X3 := r(y=3)[1].v;
END;
A21 := -1.8;
A22 := -.333;
A23 := 1.13;
B2 := -3.333;
N2 := 1000;
M2 := 3;
mX2 := DISTRIBUTE(tm.Random(N2, M2, 1.0, 2), wi_id);
X2 := pbbConverted.MatrixToNF(mX2);
sX2 := SORT(mX2, wi_id, x, LOCAL);
gX2 := GROUP(sX2, wi_id, x);
cX2 := ROLLUP(gX2,  GROUP, makeComposite2(LEFT, ROWS(LEFT)));
Layout_Cell makeY2(compX2 X) := TRANSFORM
  SELF.x := X.id;
  SELF.y := 1;
  SELF.wi_id := X.wi;
  SELF.v := A21* X.X1 + A22 * X.X2 + A23 * X.X3 + B2 + Noise(1000);
END;
mY2 := PROJECT(cX2, makeY2(LEFT));
Y2 := pbbConverted.MatrixToNF(mY2);

// Create a myriad regression with both sets of inputs
lr := LROLS.OLS(X2+X1, Y1+Y2);

// Test analytic functions starting with the most independent attributes, and working up
// from there

// Test output format
TestRec := RECORD
  STRING32 testname;
  UNSIGNED errors;
  STRING   details;
END;
// TEST01 -- Betas

// Make sure our Betas are retrieved properly
betas := lr.Betas();
betas1 := betas(wi=1);
betas2 := betas(wi=2);
details1_1 := 'B0: ' + betas1[1].value + ', B1: ' + betas1[2].value;
details1_2 := 'B0:' + betas2[1].value + ', B1: ' + betas2[2].value + ', B2: ' + 
                          betas2[3].value + ', B3: ' + betas2[4].value;

errors1_1  := 0;
errors1_2  := 0;
                 
test1_1 := DATASET([{'TEST1_1 -- Betas(1)', errors1_1, details1_1}], TestRec);
test1_2 := DATASET([{'TEST1_2 -- Betas(2)', errors1_2, details1_2}], TestRec);

INTEGER test(REAL8 v, REAL8 expected) := FUNCTION
 err := IF(ABS(v - expected) < epsilon, 0, 1);
 return err;
END;

// TEST02 -- T Distribution

tdist1 := lr.TDistribution(10);
cumT1 := tdist1.cumulative(2.0);
tdist2 := lr.TDistribution(30);
cumT2 := tdist2.cumulative(2.0);
expCumT1 := 0.96330598261462974;
expCumT2 := 0.97268747751850837;
tval1 := tdist1.Ntile(expCumT1);
tval2 := tdist2.Ntile(expCumT2);
expTval1 := 2.0;
expTval2 := 2.0;
errors2_1 := test(cumT1, expCumT1);
errors2_2 := test(cumT2, expCumT2);
errors2_3 := test(tval1, expTval1);
errors2_4 := test(tval2, expTval2);
details2_1 := 'Cumulative T: ' + cumT1 + ', Expected: ' + expCumT1;
details2_2 := 'Cumulative T: ' + cumT2 + ', Expected: ' + expCumT2;
details2_3 := 'T value: ' + tval1 + ', Expected: ' + expTval1;
details2_4 := 'T value: ' + tval2 + ', Expected: ' + expTval2;
test2_1 := DATASET([{'TEST2_1 -- T-Distr CDF(1)', errors2_1, details2_1}], TestRec);
test2_2 := DATASET([{'TEST2_2 -- T-Distr CDF(2)', errors2_2, details2_2}], TestRec);
test2_3 := DATASET([{'TEST2_3 -- T-Distr Inv CDF(1)', errors2_3, details2_3}], TestRec);
test2_4 := DATASET([{'TEST2_4 -- T-Distr Inv CDF(2)', errors2_4, details2_4}], TestRec);

// TEST03 -- F Distribution
fdist1 := lr.FDistribution(5, 10);
fdist2 := lr.FDistribution(4, 29);
cumF1 := fdist1.cumulative(2.0);
cumF2 := fdist2.cumulative(2.0);
expCumF1 := 0.83580504910026132;
expCumF2 := 0.87912522538095683;
errors3_1 := test(cumF1, expCumF1);
errors3_2 := test(cumF2, expCumF2);
details3_1 := 'Cumulative F: ' + cumF1 + ', Expected: ' + expCumF1;
details3_2 := 'Cumulative F: ' + cumF2 + ', Expected: ' + expCumF2;
nd := lr.NormalDistribution;
details3_3 := 'Density F(.5, 1, 2, [5,10]): ' + fdist1.Density(.5) + ', ' + fdist1.Density(1) + ', ' + fdist1.Density(2);
errors3_3 := 0;
test3_1 := DATASET([{'TEST3_1 -- F-Distr CDF(1)', errors3_1, details3_1}], TestRec);
test3_2 := DATASET([{'TEST3_2 -- F-Distr CDF(2)', errors3_2, details3_2}], TestRec);
test3_3 := DATASET([{'TEST3_3 -- F-Distr PDF(1)', errors3_3, details3_3}], TestRec);

// TEST04 -- R Squared
Rsq := lr.RSquared;
Rsq1 := Rsq(wi=1)[1].RSquared;
Rsq2 := Rsq(wi=2)[1].RSquared;
details4_1 := 'Rsq: ' + Rsq1;
details4_2 := 'Rsq: ' + Rsq2;
errors4_1 := 0;
errors4_2 := 0;
test4_1 := DATASET([{'TEST4_1 -- R Squared(1)', errors4_1, details4_1}], TestRec);
test4_2 := DATASET([{'TEST4_2 -- R Squared(2)', errors4_2, details4_2}], TestRec);

// TEST05 -- ANOVA
an := lr.Anova;
an1 := an(wi=1)[1];
an2 := an(wi=2)[1];
totalSS1 := an1.Total_SS;
modelSS1 := an1.Model_SS;
errorSS1 := an1.Error_SS;
SS1check := modelSS1 + errorSS1;
totalSS2 := an2.Total_SS;
modelSS2 := an2.Model_SS;
errorSS2 := an2.Error_SS;
SS2check := modelSS2 + errorSS2;
details5_1 := 'totalSS: ' + totalSS1 + ', modelSS: ' + modelSS1 + ', errorSS: ' + errorSS1;
details5_2 := 'totalSS: ' + totalSS2 + ', modelSS: ' + modelSS2 + ', errorSS: ' + errorSS2;
errors5_1 := IF(SS1check != totalSS1, 1, 0);
errors5_2 := IF(SS2check != totalSS2, 1, 0);
test5_1 := DATASET([{'TEST5_1 -- ANOVA Sum of Sq(1)', errors5_1, details5_1}], TestRec);
test5_2 := DATASET([{'TEST5_2 -- ANOVA Sum of Sq(2)', errors5_2, details5_2}], TestRec);
totalDF1 := an1.Total_DF;
modelDF1 := an1.Model_DF;
errorDF1 := an1.Error_DF;
DF1check := modelDF1 + errorDF1;
totalDF2 := an2.Total_DF;
modelDF2 := an2.Model_DF;
errorDF2 := an2.Error_DF;
DF2check := modelDF2 + errorDF2;
details5_3 := 'totalDF: ' + totalDF1 + ', modelDF: ' + modelDF1 + ', errorDF: ' + errorDF1;
details5_4 := 'totalDF: ' + totalDF2 + ', modelDF: ' + modelDF2 + ', errorDF: ' + errorDF2;
errors5_3 := IF(DF1check != totalDF1, 1, 0);
errors5_4 := IF(DF2check != totalDF2, 1, 0);
test5_3 := DATASET([{'TEST5_3 -- ANOVA DF(1)', errors5_3, details5_3}], TestRec);
test5_4 := DATASET([{'TEST5_2 -- ANOVA DF(2)', errors5_4, details5_4}], TestRec);
modelMS1 := an1.Model_MS;
errorMS1 := an1.Error_MS;
modelF1  := an1.Model_F;
details5_5 := 'modelMS: ' + modelMS1 + ', errorMS: ' + errorMS1 + ', modelF: ' + modelF1;
errors5_5 := 0;
modelMS2 := an2.Model_MS;
errorMS2 := an2.Error_MS;
modelF2  := an2.Model_F;
details5_6 := 'modelMS: ' + modelMS2 + ', errorMS: ' + errorMS2 + ', modelF: ' + modelF2;
errors5_6 := 0;
test5_5 := DATASET([{'TEST5_5 -- ANOVA Mean Sq(1)', errors5_5, details5_5}], TestRec);
test5_6 := DATASET([{'TEST5_6 -- ANOVA Mean Sq(2)', errors5_6, details5_6}], TestRec);

// TEST06 -- Adjusted R Squared
aRsq := lr.AdjRSquared;
aRsq1 := aRsq(wi=1)[1].RSquared;
aRsq2 := aRsq(wi=2)[1].RSquared;
details6_1 := 'adjRsq: ' + aRsq1;
details6_2 := 'adjRsq: ' + aRsq2;
errors6_1 := 0;
errors6_2 := 0;
test6_1 := DATASET([{'TEST6_1 -- Adjusted R Squared(1)', errors6_1, details6_1}], TestRec);
test6_2 := DATASET([{'TEST6_2 -- AdjustedR Squared(2)', errors6_2, details6_2}], TestRec);

// TEST07 -- Coef_covar
coco := lr.Coef_Covar;
coco1 := coco(wi=1);
coco2 := coco(wi=2);
cocon1 := COUNT(coco1);
cocon2 := COUNT(coco2);
diag1 := coco1(id=number);
diag2 := coco2(id=number);
details7_1 := 'Cells: ' + cocon1 + ', var(0): ' + diag1[1].value + ', var(1): ' + diag1[2].value;
details7_2 := 'Cells: ' + cocon2 + ', var(0): ' + diag2[1].value + ', var(1): ' + diag2[2].value
               + ', var(2): ' + diag2[3].value + ', var(3): ' + diag2[4].value;
errors7_1  := IF(cocon1 != 4, 1, 0);
errors7_2  := IF(cocon2 != 16, 1, 0);
test7_1 := DATASET([{'TEST7_1 -- Coefficient Covar(1)', errors7_1, details7_1}], TestRec);
test7_2 := DATASET([{'TEST7_2 -- Coefficient Covar(2)', errors7_2, details7_2}], TestRec);

// TEST08 -- Standard Error
se := lr.SE;
se1 := se(wi=1);
se2 := se(wi=2);
se1count := COUNT(se1);
se2count := COUNT(se2);
details8_1 := 'SE(0): ' + se1[1].value + ', SE(1): ' + se1[2].value;
details8_2 := 'SE(0): ' + se2[1].value + ', SE(1): ' + se2[2].value + ', SE(2): ' 
                        + se2[3].value + ', SE(3): ' + se2[4].value;
errors8_1 := IF(se1count != M1+1, 1, 0);
errors8_2 := IF(se2count != M2+1, 1, 0);
test8_1 := DATASET([{'TEST8_1 -- Standard Error(1)', errors8_1, details8_1}], TestRec);
test8_2 := DATASET([{'TEST8_2 -- Standard Error(2)', errors8_2, details8_2}], TestRec);

// TEST09 -- T Statistic
tstat := lr.TStat;
tstat1 := tstat(wi=1);
tstat2 := tstat(wi=2);
details9_1 := 'TStat(0): ' + tstat1[1].value + ', TStat(1): ' + tstat1[2].value;
details9_2 := 'TStat(0): ' + tstat2[1].value + ', TStat(1): ' + tstat2[2].value
          + ', TStat(2): ' + tstat2[3].value + ', TStat(3): ' + tstat2[4].value;
errors9_1 := 0;
errors9_2 := 0;
test9_1 := DATASET([{'TEST9_1 -- T-Statistic(1)', errors9_1, details9_1}], TestRec);
test9_2 := DATASET([{'TEST9_2 -- T-Statistic(2)', errors9_2, details9_2}], TestRec);

// TEST10 -- P-val
p_val := lr.pVal;
p_val1 := lr.pVal(wi=1);
p_val2 := lr.pVal(wi=2);
details10_1 := 'Pval(0): ' + p_val1[1].value + ', Pval(1): ' + p_val1[2].value;
errors10_1 := 0;
details10_2 := 'Pval(0): ' + p_val2[1].value + ', Pval(1): ' + p_val2[2].value 
                           + ', Pval(2): ' + p_val2[3].value + ', Pval(3): ' + p_val2[4].value;
errors10_2 := 0;
test10_1 := DATASET([{'TEST10_1 -- Pval(1)', errors10_1, details10_1}], TestRec);
test10_2 := DATASET([{'TEST10_2 -- Pval(2)', errors10_2, details10_2}], TestRec);

// TEST11 -- AIC
aic := lr.AIC;
aic1 := aic(wi=1)[1].AIC;
aic2 := aic(wi=2)[1].AIC;
details11_1 := 'AIC: ' + aic1;
details11_2 := 'AIC: ' + aic2;
errors11_1 := 0;
errors11_2 := 0;
test11_1 := DATASET([{'TEST11_1 -- AIC(1)', errors11_1, details11_1}], TestRec);
test11_2 := DATASET([{'TEST11_2 -- AIC(2)', errors11_2, details11_2}], TestRec);

// TEST12 -- Confidence Interval
confint := lr.ConfInt(95); // 95% confidence
confint1 := confint(wi=1);
confint2 := confint(wi=2);
details12_1 := 'CI(0): ' +  confint1[1].LowerInt + ' - ' + confint1[1].UpperInt
              + ', CI(1): ' + confint1[2].LowerInt + ' - ' + confint1[2].UpperInt;
details12_2 := 'CI(0): ' +  confint2[1].LowerInt + ' - ' + confint2[1].UpperInt
              + ', CI(1): ' + confint2[2].LowerInt + ' - ' + confint2[2].UpperInt
              + ', CI(2): ' + confint2[3].LowerInt + ' - ' + confint2[3].UpperInt
              + ', CI(3): ' + confint2[4].LowerInt + ' - ' + confint2[4].UpperInt;
errors12_1 := 0;
errors12_2 := 0;
test12_1 := DATASET([{'TEST12_1 -- ConfInt(1)', errors12_1, details12_1}], TestRec);
test12_2 := DATASET([{'TEST12_2 -- ConfInt(2)', errors12_2, details12_2}], TestRec);

// TEST13 -- FTest
ft := lr.FTest;
ft1 := ft(wi=1)[1];
ft2 := ft(wi=2)[1];
details13_1 := 'Model_F: ' + ft1.Model_F + ', pValue: ' + ft1.Pvalue;
details13_2 := 'Model_F: ' + ft2.Model_F + ', pValue: ' + ft2.Pvalue;
errors13_1 := 0;
errors13_2 := 0;
test13_1 := DATASET([{'TEST13_1 -- F-Test(1)', errors13_1, details13_1}], TestRec);
test13_2 := DATASET([{'TEST13_2 -- F-Test(2)', errors13_2, details13_2}], TestRec);

summary := test1_1 + test1_2
            + test2_1 + test2_2 + test2_3 + test2_4
            + test3_1 + test3_2 + test3_3
            + test4_1 + test4_2 
            + test5_1 + test5_2 + test5_3 + test5_4 + test5_5 + test5_6
            + test6_1 + test6_2
            + test7_1 + test7_2
            + test8_1 + test8_2
            + test9_1 + test9_2
            + test10_1 + test10_2
            + test11_1 + test11_2
            + test12_1 + test12_2
            + test13_1 + test13_2;

EXPORT OLSAnalytics := summary;

