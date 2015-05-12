#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <msclr/marshal.h>
#include "svm.h"

using namespace System;
using namespace Runtime::InteropServices;
using namespace msclr::interop;

#define COPY_MANAGED_ARRAY_TO_NATIVE(type,n,NAME_NAT,NAME_MAN) { auto count = (n); native.NAME_NAT = new type[n]; pin_ptr<type> p = &model.NAME_MAN[0]; memcpy(native.NAME_NAT, p, (n) * sizeof(type)); }

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

		/// <summary>
		/// Number of training data entries.
		/// </summary>
		property int Count {
			int get() {
				return y->Length;
			}
		}

		Problem(array<array<Node>^>^ trainingVectors, array<double>^ targetValues) : x(trainingVectors), y(targetValues)
		{
			if (x->Length != y->Length)
			{
				throw gcnew ArgumentOutOfRangeException("Their need to be as many target values as training vectors.");
			}
		}

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
				pin_ptr<Node> pr = &r[0];
				memcpy(pr, row, count * sizeof(svm_node));
				x[i] = r;
			}

			y = gcnew array<double>(l);
			pin_ptr<double> py = &y[0];
			memcpy(py, problem.y, l * sizeof(double));
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

		/// <summary>
		/// Gets SVM type of this model.
		/// </summary>
		property SvmType Type {
			SvmType get() {
				return Param.SvmType;
			}
		}

	internal:

		Model(const svm_model* native)
		{
			auto k = native->nr_class;

			// param
			Param = Parameter(native->param);

			// nr_class
			NrClass = k;

			// l
			l = native->l;

			// nSV
			nSV = gcnew array<int>(k);
			Marshal::Copy((IntPtr)native->nSV, nSV, 0, k);

			// label
			Label = gcnew array<int>(k);
			Marshal::Copy((IntPtr)native->label, Label, 0, k);

			// free_sv
			FreeSv = native->free_sv;

			// sv_indices
			SvIndices = gcnew array<int>(l);
			Marshal::Copy((IntPtr)native->sv_indices, SvIndices, 0, l);

			// SV
			SV = gcnew array<array<Node>^>(l);
			for (auto i = 0; i < l; i++)
			{
				auto count = 0;
				svm_node* p = native->SV[i];
				while ((p++)->index != -1) ++count;

				auto row = gcnew array<Node>(count);
				pin_ptr<Node> pRow = &row[0];
				memcpy(pRow, native->SV[i], count * sizeof(svm_node));
				SV[i] = row;
			}

			// sv_coeff
			SvCoef = gcnew array<array<double>^>(k-1);
			for (auto i = 0; i < k - 1; i++)
			{
				SvCoef[i] = gcnew array<double>(l);
				Marshal::Copy((IntPtr)native->sv_coef[i], SvCoef[i], 0, l);
			}

			// rho
			auto count = k*(k - 1) / 2;
			Rho = gcnew array<double>(count);
			Marshal::Copy((IntPtr)native->rho, Rho, 0, count);

			// probA/probB
			if (Param.Probability != 0)
			{
				ProbA = gcnew array<double>(count);
				Marshal::Copy((IntPtr)native->probA, ProbA, 0, count);

				ProbB = gcnew array<double>(count);
				Marshal::Copy((IntPtr)native->probB, ProbB, 0, count);
			}
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

					// debug code
					//svm_save_model("C:/Data/test_r1.txt", r);
					//svm_save_model("C:/Data/test_r2.txt", &Convert(Model(r)));

					// (4) convert result svm_model to managed Model
					auto result = Model(r);
					FreeNativeModel(r, false);
					return result;
				}
				finally
				{
					FreeProblem(&arg_problem);
					svm_destroy_param(&arg_parameter);
				}
			}
			
			/// <summary>
			/// This function conducts cross validation. Data are separated to
			/// NrFold folds. Under given parameters, sequentially each fold is
			/// validated using the model from training the remaining. Predicted
			/// labels (of all prob's instances) in the validation process are
			/// stored in the array called target.
			/// The format of problem is same as that for Train().
			/// </summary>
			static array<double>^ CrossValidation(Problem problem, Parameter parameter, int nrFold)
			{
				svm_problem arg_problem;
				svm_parameter arg_parameter;
				double* target = NULL;

				try
				{
					arg_problem = Convert(problem);
					arg_parameter = Convert(parameter);

					auto l = problem.Count;
					auto target = new double[l];
					svm_cross_validation(&arg_problem, &arg_parameter, nrFold, target);

					auto result = gcnew array<double>(l);
					for (auto i = 0; i < l; i++) result[i] = target[i];
					return result;
				}
				finally
				{
					FreeProblem(&arg_problem);
					if (target != NULL) free(target);
				}
			}

			/// <summary>
			/// This function does classification or regression on a test vector x
			/// given a model.
			/// For a classification model, the predicted class for x is returned.
			/// For a regression model, the function value of x calculated using
			/// the model is returned.For an one - class model, +1 or -1 is
			/// returned.
			/// </summary>
			static double Predict(Model model, array<Node>^ x)
			{
				svm_node* nodes = NULL;
				svm_model nativeModel;
				try
				{
					nodes = new svm_node[x->Length + 1];
					pin_ptr<Node> p = &x[0];
					memcpy(nodes, p, x->Length * sizeof(svm_node));
					nodes[x->Length] = { -1, 0 };
					nativeModel = Convert(model);
					return svm_predict(&nativeModel, nodes);
				}
				finally
				{
					if (nodes) delete[] nodes;
					FreeNativeModel(&nativeModel, true);
				}
			}

			/// <summary>
			/// This function gives decision values on a test vector x given a
			/// model, and returns the predicted label (classification) or
			/// the function value (regression).
			/// For a classification model with NrClass classes, this function
			/// gives NrClass*(NrClass - 1)/2 decision values in the array
			/// decValues. The order is label[0] vs.label[1], ...,
			/// label[0] vs.label[NrClass - 1], label[1] vs.label[2], ...,
			/// label[NrClass - 2] vs.label[NrClass - 1]. The returned value is
			/// the predicted class for x. Note that when NrClass=1, this
			/// function does not give any decision value.
			/// For a regression model, decValues[0] and the returned value are
			/// both the function value of x calculated using the model. For a
			/// one-class model, decValues[0] is the decision value of x, while
			/// the returned value is +1/-1.
			/// </summary>
			static double PredictValues(Model model, array<Node>^ x, array<double>^ decValues)
			{
				svm_node* nodes = NULL;
				svm_model nativeModel;
				try
				{
					nodes = new svm_node[x->Length + 1];
					pin_ptr<Node> p = &x[0];
					memcpy(nodes, p, x->Length * sizeof(svm_node));
					nodes[x->Length] = { -1, 0 };
					nativeModel = Convert(model);
					pin_ptr<double> pDecValues = &decValues[0];
					return svm_predict_values(&nativeModel, nodes, pDecValues);
				}
				finally
				{
					if (nodes) delete[] nodes;
					FreeNativeModel(&nativeModel, true);
				}
			}

			/// <summary>
			/// This function does classification or regression on a test vector x
			/// given a model with probability information.
			/// For a classification model with probability information, this
			/// function gives NrClass probability estimates in the array
			/// probEstimates. The class with the highest probability is
			/// returned. For regression/one-class SVM, the array probEstimates
			/// is unchanged and the returned value is the same as that of
			/// Predict.
			/// </summary>
			static double PredictProbability(Model model, array<Node>^ x, array<double>^ probEstimates)
			{
				svm_node* nodes = NULL;
				svm_model nativeModel;
				try
				{
					nodes = new svm_node[x->Length + 1];
					pin_ptr<Node> p = &x[0];
					memcpy(nodes, p, x->Length * sizeof(svm_node));
					nodes[x->Length] = { -1, 0 };
					nativeModel = Convert(model);
					pin_ptr<double> pProbEstimates = &probEstimates[0];
					return svm_predict_probability(&nativeModel, nodes, pProbEstimates);
				}
				finally
				{
					if (nodes) delete[] nodes;
					FreeNativeModel(&nativeModel, true);
				}
			}

			/// <summary>
			/// This function saves a model to a file.
			/// </summary>
			static void SaveModel(String^ fileName, Model model)
			{
				auto context = gcnew marshal_context();
				const char* nativeFileName = context->marshal_as<const char*>(fileName);

				auto native_model = Convert(model);
				auto err = svm_save_model(nativeFileName, &native_model);

				if (err != 0) throw gcnew Exception("SaveModel failed with error code " + err + ".");
			}

			/// <summary>
			/// This function loads a model from file.
			/// </summary>
			static Model LoadModel(String^ fileName)
			{
				auto context = gcnew marshal_context();
				const char* nativeFileName = context->marshal_as<const char*>(fileName);

				auto native_model = svm_load_model(nativeFileName);
				if (native_model == NULL) throw gcnew Exception("LoadModel failed to load model from " + fileName + ".");

				return Model(native_model);
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
					FreeProblem(&arg_problem);
				}
			}

			/// <summary>
			/// This function checks whether the model contains required information
			/// to do probability estimates. If so, it returns true. Otherwise, false
			/// is returned. This function should be called before calling
			/// GetSvrProbability and PredictProbability.
			/// </summary>
			static bool CheckProbabilityModel(Model model)
			{
				svm_model nativeModel;
				try
				{
					nativeModel = Convert(model);
					auto r = svm_check_probability_model(&nativeModel);
					return r == 1;
				}
				finally
				{
					FreeNativeModel(&nativeModel, true);
				}
				
			}

		private:

			static svm_problem Convert(Problem problem)
			{
				svm_problem result;

				auto l = result.l = problem.y->Length;
				pin_ptr<double> yPinned = &problem.y[0];

				result.y = new double[l];
				memcpy(result.y, yPinned, l * sizeof(double));

				result.x = new svm_node*[l];
				for (int i = 0; i < l; i++)
				{
					auto lengthRow = problem.x[i]->Length;
					auto pRow = result.x[i] = new svm_node[lengthRow + 1];
					pin_ptr<Node> pinnedRow = &problem.x[i][0];
					memcpy(pRow, pinnedRow, lengthRow * sizeof(svm_node));
					pRow[lengthRow].index = -1; pRow[lengthRow].value = 0.0; // add delimiter
				}

				return result;
			}

			static svm_node Convert(const Node node)
			{
				svm_node x;
				x.index = node.Index;
				x.value = node.Value;
				return x;
			}
			
			static svm_parameter Convert(Parameter parameter)
			{
				svm_parameter result;
				result.svm_type = (int)parameter.SvmType;
				result.kernel_type = (int)parameter.KernelType;
				result.degree = parameter.Degree;
				result.gamma = parameter.Gamma;
				result.coef0 = parameter.Coef0;
				result.cache_size = parameter.CacheSize;
				result.eps = parameter.Eps;
				result.C = parameter.C;
				result.nr_weight = parameter.Weight ? parameter.Weight->Length : 0;
				if (parameter.WeightLabel && parameter.WeightLabel->Length)
				{
					auto count = parameter.WeightLabel->Length;

					result.weight_label = new int[count];
					pin_ptr<int> pinnedWeightLabel = &parameter.WeightLabel[0];
					memcpy(result.weight_label, pinnedWeightLabel, count * sizeof(int));

					result.weight = new double[count];
					pin_ptr<double> pinnedWeight = &parameter.Weight[0];
					memcpy(result.weight, pinnedWeight, count * sizeof(double));
				}
				else
				{
					result.weight_label = NULL;
					result.weight = NULL;
				}
				result.nu = parameter.Nu;
				result.p = parameter.p;
				result.shrinking = parameter.Shrinking;
				result.probability = parameter.Probability;
				
				return result;
			}

			static svm_model Convert(Model model)
			{
				auto k = model.NrClass;
				auto l = model.l;

				svm_model native;

				// param
				native.param = Convert(model.Param);

				// nr_class
				native.nr_class = model.NrClass;

				// l
				native.l = l;

				// nSV
				COPY_MANAGED_ARRAY_TO_NATIVE(int, k, nSV, nSV)

				// label
				COPY_MANAGED_ARRAY_TO_NATIVE(int, k, label, Label)

				// free_sv
				native.free_sv = model.FreeSv;

				// sv_indices
				COPY_MANAGED_ARRAY_TO_NATIVE(int, l, sv_indices, SvIndices)

				// SV
				native.SV = new svm_node*[l];
				for (auto i = 0; i < l; i++)
				{
					auto r = model.SV[i];
					pin_ptr<Node> p = &r[0];
					native.SV[i] = new svm_node[r->Length + 1];
					memcpy(native.SV[i], p, r->Length * sizeof(svm_node));
					native.SV[i][r->Length] = { -1, 0 };
				}

				// sv_coeff
				native.sv_coef = new double*[k - 1];
				for (auto i = 0; i < k - 1; i++)
				{
					COPY_MANAGED_ARRAY_TO_NATIVE(double, l, sv_coef[i], SvCoef[i])
				}

				// rho
				COPY_MANAGED_ARRAY_TO_NATIVE(double, k*(k - 1) / 2, rho, Rho)

				if (model.Param.Probability != 0)
				{
					// probA
					COPY_MANAGED_ARRAY_TO_NATIVE(double, k*(k - 1) / 2, probA, ProbA)
					// probB
					COPY_MANAGED_ARRAY_TO_NATIVE(double, k*(k - 1) / 2, probB, ProbB)
				}
				else
				{
					native.probA = native.probB = NULL;
				}

				return native;
			}

			static void FreeProblem(svm_problem* problem)
			{
				if (problem->y)
				{
					delete[] problem->y;
					problem->y = NULL;
				}
				if (problem->x)
				{
					for (int i = 0; i < problem->l; i++)
					{
						delete[] problem->x[i];
						problem->x[i] = NULL;
					}
					delete[] problem->x;
					problem->x = NULL;
				}
			}

			static void FreeNativeModel(svm_model* native, bool deleteInnerSVs)
			{
				#define Clean(param) { if (native->param) delete[] native->param; native->param = NULL; }

				Clean(param.weight);
				Clean(param.weight_label);

				Clean(nSV);
				Clean(label);
				Clean(sv_indices);

				if (native->SV)
				{
					if (deleteInnerSVs)
					{
						for (auto i = 0; i < native->l; i++)
						{
							delete[] native->SV[i];
							native->SV[i] = NULL;
						}
					}

					delete[] native->SV;
					native->SV = NULL;
				}

				if (native->sv_coef)
				{
					for (auto i = 0; i < native->nr_class - 1; i++) delete[] native->sv_coef[i];
					delete[] native->sv_coef;
				}

				Clean(rho);
				Clean(probA);
				Clean(probB);
			}
	};
}
