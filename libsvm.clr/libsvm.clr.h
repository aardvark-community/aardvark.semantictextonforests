#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "svm.h"

using namespace System;

namespace LibSvm {

	public enum class SvmType
	{
		C_SVC = ::C_SVC,
		NU_SVC = ::NU_SVC,
		ONE_CLASS = ::ONE_CLASS,
		EPSILON_SVR = ::EPSILON_SVR,
		NU_SVR = ::NU_SVR
	};

	public enum class KernelType
	{
		LINEAR = ::LINEAR,
		POLY = ::POLY,
		RBF = ::RBF,
		SIGMOID = ::SIGMOID,
		PRECOMPUTED = ::PRECOMPUTED
	};

	public value struct Node
	{
		int Index;
		double Value;

		Node(int index, double value) : Index(index), Value(value) { }

	internal:

		Node(const svm_node& node) : Index(node.index), Value(node.value) { }
	};

	public value struct Problem
	{
		array<array<Node>^>^ x;
		array<double>^ y;

		Problem(array<array<Node>^>^ x, array<double>^ y) : x(x), y(y) { }

	internal:

		Problem(const svm_problem& problem)
		{
			auto l = problem.l;

			x = gcnew array<array<Node>^>(l);
			for (auto i = 0; i < l; i++)
			{
				auto row = problem.x[i];
				auto count = 0;
				while (row[count].index != -1) count++;
				auto r = gcnew array<Node>(count);
				for (auto j = 0; j < count; j++) r[j] = Node(row[j]);
				x[i] = r;
			}

			y = gcnew array<double>(l);
			for (int i = 0; i < l; i++)
			{
				y[i] = problem.y[i];
			}
		}
	};

	public value struct Parameter
	{
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
		double p;

		/// <summary>use the shrinking heuristics</summary>
		int Shrinking;

		/// <summary>do probability estimates</summary>
		int Probability;

	internal:

		Parameter(const svm_parameter& native)
		{
			SvmType = (LibSvm::SvmType)native.svm_type;
			KernelType = (LibSvm::KernelType)native.kernel_type;
			Degree = native.degree;
			Gamma = native.gamma;
			Coef0 = native.coef0;
			CacheSize = native.cache_size;
			Eps = native.eps;
			C = native.C;
			WeightLabel = gcnew array<int>(native.nr_weight);
			Weight = gcnew array<double>(native.nr_weight);
			Nu = native.nu;
			p = native.p;
			Shrinking = native.shrinking;
			Probability = native.probability;

			for (auto i = 0; i < native.nr_weight; i++) WeightLabel[i] = native.weight_label[i];
			for (auto i = 0; i < native.nr_weight; i++) Weight[i] = native.weight[i];
		}
	};

	public value struct Model
	{
		/// <summary>parameter</summary>
		Parameter Param;
		
		/// <summary>number of classes, = 2 in regression/one class svm</summary>
		int NrClass;
		
		/// <summary>total #SV</summary>
		int l;
		
		/// <summary>SVs (SV[l])</summary>
		array<array<Node>^>^ SV;
		
		/// <summary>coefficients for SVs in decision functions (sv_coef[k-1][l])</summary>
		array<array<double>^>^ SvCoef;
		
		/// <summary>constants in decision functions (rho[k*(k-1)/2])</summary>
		array<double>^ Rho;
		
		/// <summary>pairwise probability information</summary>
		array<double>^ ProbA;

		/// <summary></summary>
		array<double>^ ProbB;

		/// <summary>sv_indices[0,...,nSV-1] are values in [1,...,num_training_data] to indicate SVs in the training set</summary>
		array<int>^ SvIndices;

		/* for classification only */

		/// <summary>label of each class (label[k])</summary>
		array<int>^ Label;
		
		/// <summary>
		/// number of SVs for each class (nSV[k])
		/// nSV[0] + nSV[1] + ... + nSV[k-1] = l
		/// XXX
		/// </summary>
		array<int>^ nSV;
					
		/// <summary>
		/// 1 if svm_model is created by svm_load_model
		/// 0 if svm_model is created by svm_train
		/// </summary>
		int FreeSv;

	internal:

		Model(const svm_model* native)
		{
			auto k = native->nr_class;

			Param = Parameter(native->param);
			NrClass = k;
			l = native->l;

			SV = gcnew array<array<Node>^>(l);
			// TODO: init inner arrays

			SvCoef = gcnew array<array<double>^>(k);
			for (auto i = 0; i < k; i++)
			{
				SvCoef[i] = gcnew array<double>(l);
				for (auto j = 0; j < l; j++) SvCoef[i][j] = native->sv_coef[i][j];
			}

			// TODO: Rho
			// TODO: ProbA
			// TODO: ProbB
			// TODO: SvIndices

			Label = gcnew array<int>(k);
			for (auto i = 0; i < k; i++) Label[i] = native->label[i];

			nSV = gcnew array<int>(k);
			for (auto i = 0; i < k; i++) nSV[i] = native->nSV[i];

			FreeSv = native->free_sv;
		}
	};

	public ref class Svm abstract sealed
	{
		public:

			/// <summary>
			/// This function constructs and returns an SVM model according
			/// to the given training data and parameters.
			/// </summary>
			static Model Train(Problem problem, Parameter parameter)
			{
				svm_problem arg_problem;
				svm_parameter arg_parameter;

				try
				{
					// (1) convert managed Problem to svm_problem
					arg_problem = Convert(problem);

					// (2) convert managed Parameter to svm_parameter
					arg_parameter = Convert(parameter);

					// (3) call actual function
					auto r = svm_train(&arg_problem, &arg_parameter);

					// (4) convert result svm_model to managed Model
					return Model(r);
				}
				finally
				{
					FreeProblem(arg_problem);
				}
			}

			/// <summary>
			/// This function checks whether the parameters are within the feasible
			/// range of the problem. This function should be called before 
			/// Svm.Train() and Svm.CrossValidation(). It returns null if the
			/// parameters are feasible, otherwise an error message is returned.
			/// </summary>
			static String^ CheckParameter(Problem problem, Parameter parameter)
			{
				svm_problem arg_problem;
				svm_parameter arg_parameter;

				try
				{
					arg_problem = Convert(problem);
					arg_parameter = Convert(parameter);
					auto r = svm_check_parameter(&arg_problem, &arg_parameter);
					return gcnew String(r);
				}
				finally
				{
					FreeProblem(arg_problem);
				}
			}

		private:

			static svm_problem Convert(Problem problem)
			{
				svm_problem result;

				int l = problem.y->Length;
				pin_ptr<double> yPinned = &problem.y[0];

				result.l = l;
				result.y = yPinned;
				result.x = (svm_node**)malloc(l * sizeof(void*));

				for (int i = 0; i < l; i++)
				{
					auto lengthRow = problem.x[i]->Length;
					auto pRow = (svm_node*)malloc((lengthRow + 1) * sizeof(svm_node)); // +1 for delimiter
					pin_ptr<array<Node>^> pinnedRow = &problem.x[i];
					memcpy(pRow, pinnedRow, lengthRow * sizeof(svm_node));
					pRow[lengthRow].index = -1; pRow[lengthRow].value = 0.0; // add delimiter
					result.x[i] = pRow;
				}

				return result;
			}

			static svm_parameter Convert(Parameter parameter)
			{
				pin_ptr<int> pinnedWeightLabel;
				pin_ptr<double> pinnedWeight;
				if (parameter.WeightLabel->Length)
				{
					pinnedWeightLabel = &parameter.WeightLabel[0];
					pinnedWeight = &parameter.Weight[0];
				}

				svm_parameter result;
				result.svm_type = (int)parameter.SvmType;
				result.kernel_type = (int)parameter.KernelType;
				result.degree = parameter.Degree;
				result.gamma = parameter.Gamma;
				result.coef0 = parameter.Coef0;
				result.cache_size = parameter.CacheSize;
				result.eps = parameter.Eps;
				result.C = parameter.C;
				result.nr_weight = parameter.Weight->Length;
				result.weight_label = pinnedWeightLabel;
				result.weight = pinnedWeight;
				result.nu = parameter.Nu;
				result.p = parameter.p;
				result.shrinking = parameter.Shrinking;
				result.probability = parameter.Probability;
				return result;
			}

			static void FreeProblem(svm_problem problem)
			{
				if (problem.x == NULL) return;

				for (int i = 0; i < problem.l; i++)
				{
					free(problem.x[i]);
				}

				free(problem.x);
			}
	};
}
