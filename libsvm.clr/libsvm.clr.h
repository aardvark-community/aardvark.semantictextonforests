// libsvm.clr.h

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "svm.h"

using namespace System;

namespace libsvmclr {

	public enum class SvmType  { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };
	public enum class KernelType { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };

	public value class Node
	{
	public:
		int Index;
		double Value;
	};

	public value class Problem
	{
	public:
		array<double>^ y;
		array<array<Node>^>^ x;
	};

	/// <summary>
	/// Parameter.
	/// </summary>
	public value class Parameter
	{
	public:
		/// <summary></summary>
		SvmType SvmType;

		/// <summary></summary>
		KernelType KernelType;

		/// <summary>for poly</summary>
		int Degree;

		/// <summary>for poly/rbf/sigmoid</summary>
		double Gamma;

		/// <summary>for poly/sigmoid</summary>
		double Coef0;

						/* these are for training only */

		/// <summary>in MB</summary>
		double CacheSize;

		/// <summary>stopping criteria</summary>
		double Eps;

		/// <summary>for C_SVC, EPSILON_SVR and NU_SVR</summary>
		double C;

		/// <summary></summary>
		array<int>^ WeightLabel;

		/// <summary></summary>
		array<double>^ Weight;


		/// <summary>for NU_SVC, ONE_CLASS, and NU_SVR</summary>
		double Nu;

		/// <summary>for EPSILON_SVR</summary>
		double P;

		/// <summary>use the shrinking heuristics</summary>
		int Shrinking;

		/// <summary>do probability estimates</summary>
		int Probability;
	};

	public ref class Model
	{
		Parameter param;	/* parameter */
		int NrClass;		/* number of classes, = 2 in regression/one class svm */
		int L;			/* total #SV */
		struct svm_node **SV;		/* SVs (SV[l]) */
		double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
		double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
		double *probA;		/* pariwise probability information */
		double *probB;
		int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

								/* for classification only */

		int *label;		/* label of each class (label[k]) */
		int *nSV;		/* number of SVs for each class (nSV[k]) */
						/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
						/* XXX */
		int free_sv;		/* 1 if svm_model is created by svm_load_model*/
							/* 0 if svm_model is created by svm_train */
	};

	public ref class Svm abstract sealed
	{
		public:

			static Model^ Train(Problem problem, Parameter param)
			{
				svm_problem* arg_problem = NULL;
				svm_parameter* arg_param = NULL;

				try
				{
					// (1) convert managed Problem to svm_problem
					arg_problem = (svm_problem*)malloc(sizeof(svm_problem));

					int l = problem.y->Length;
					pin_ptr<double> yPinned = &problem.y[0];

					arg_problem->l = l;
					arg_problem->y = yPinned;
					arg_problem->x = (svm_node**)malloc(l * sizeof(void*));

					for (int i = 0; i < l; i++)
					{
						auto lengthRow = problem.x[i]->Length;
						auto pRow = (svm_node*)malloc((lengthRow + 1) * sizeof(svm_node)); // +1 for delimiter
						pin_ptr<array<Node>^> pinnedRow = &problem.x[i];
						memcpy(pRow, pinnedRow, lengthRow * sizeof(svm_node));
						pRow[lengthRow].index = -1; pRow[lengthRow].value = 0.0; // add delimiter
						arg_problem->x[i] = pRow;
					}

					// (2) convert managed Parameter to svm_parameter
					pin_ptr<int> pinnedWeightLabel;
					pin_ptr<double> pinnedWeight;
					if (param.WeightLabel->Length)
					{
						pinnedWeightLabel = &param.WeightLabel[0];
						pinnedWeight = &param.Weight[0];
					}
					
					arg_param = (svm_parameter*)malloc(sizeof(svm_parameter));
					arg_param->svm_type = (int)param.SvmType;
					arg_param->kernel_type = (int)param.KernelType;
					arg_param->degree = param.Degree;
					arg_param->gamma = param.Gamma;
					arg_param->coef0 = param.Coef0;
					arg_param->cache_size = param.CacheSize;
					arg_param->eps = param.Eps;
					arg_param->C = param.C;
					arg_param->nr_weight = param.Weight->Length;
					arg_param->weight_label = pinnedWeightLabel;
					arg_param->weight = pinnedWeight;
					arg_param->nu = param.Nu;
					arg_param->p = param.P;
					arg_param->shrinking = param.Shrinking;
					arg_param->probability = param.Probability;

					// (3) call actual function
					auto r = svm_train(arg_problem, arg_param);

					// (4) convert result svm_model to managed Model
					auto result = gcnew Model;
					// TODO
					return result;
				}
				finally
				{
					if (arg_problem) free(arg_problem);
					if (arg_param) free(arg_param);
				}
			}
	};
}
