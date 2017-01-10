IMPORT Std;
EXPORT Bundle := MODULE(Std.BundleBase)
 EXPORT Name := 'LinearRegressionOLS';
 EXPORT Description := 'Ordinary Least Squares Linear Regression';
 EXPORT Authors := ['John Holt', 'Roger Dev'];
 EXPORT License := 'http://www.apache.org/licenses/LICENSE-2.0';
 EXPORT Copyright := 'Copyright (C) 2017 HPCC Systems';
 EXPORT DependsOn := [ML_Core, PBblas];
 EXPORT Version := '6.2.0';
END;