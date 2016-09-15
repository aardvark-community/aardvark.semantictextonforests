#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <msclr/marshal.h>
#include "linear.h"

using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;
using namespace System::Globalization;
using namespace System::IO;
using namespace System::Runtime::InteropServices;
using namespace msclr::interop;

#define COPY_MANAGED_ARRAY_TO_NATIVE(type,n,NAME_NAT,NAME_MAN) { auto count = (n); native.NAME_NAT = new type[n]; pin_ptr<type> p = &model.NAME_MAN[0]; memcpy(native.NAME_NAT, p, (n) * sizeof(type)); }

namespace liblinear_train {
	extern struct model* main(int argc, char **argv);
}

namespace LibLinear {

	public enum class SolverType
	{
		L2R_LR = ::L2R_LR,
		L2R_L2LOSS_SVC_DUAL = ::L2R_L2LOSS_SVC_DUAL,
		L2R_L2LOSS_SVC = ::L2R_L2LOSS_SVC,
		L2R_L1LOSS_SVC_DUAL = ::L2R_L1LOSS_SVC_DUAL,
		MCSVM_CS = ::MCSVM_CS,
		L1R_L2LOSS_SVC = ::L1R_L2LOSS_SVC,
		L1R_LR = ::L1R_LR,
		L2R_LR_DUAL = ::L2R_LR_DUAL,
		L2R_L2LOSS_SVR = ::L2R_L2LOSS_SVR,
		L2R_L2LOSS_SVR_DUAL = ::L2R_L2LOSS_SVR_DUAL,
		L2R_L1LOSS_SVR_DUAL = ::L2R_L1LOSS_SVR_DUAL
	};

	public value struct Node
	{
		int Index;
		double Value;

		Node(int index, double value) : Index(index), Value(value) { }

	internal:

		Node(const feature_node& node) : Index(node.index), Value(node.value) { }
	};

	public value struct Problem
	{
		array<array<Node>^>^ x;
		array<double>^ y;
		double Bias;

		/// <summary>
		/// Number of vectors.
		/// </summary>
		property int Count {
			int get() {
				return y->Length;
			}
		}

		Problem(array<array<Node>^>^ xs, array<double>^ ys)
			: Problem(xs, ys, -1.0)
		{
		}

		Problem(array<array<Node>^>^ xs, array<double>^ ys, double bias)
			: x(xs), y(ys), Bias(bias)
		{
			if (x->Length != y->Length)
			{
				throw gcnew ArgumentOutOfRangeException("Their need to be as many target values as training vectors.");
			}
		}

	internal:

		Problem(const problem& problem)
		{
			auto l = problem.l;
			auto n = problem.n;

			x = gcnew array<array<Node>^>(l);
			for (auto i = 0; i < l; i++)
			{
				auto row = problem.x[i];
				auto r = gcnew array<Node>(n);
				pin_ptr<Node> pr = &r[0];
				memcpy(pr, row, n * sizeof(feature_node));
				x[i] = r;
			}

			y = gcnew array<double>(l);
			pin_ptr<double> py = &y[0];
			memcpy(py, problem.y, l * sizeof(double));

			Bias = problem.bias;
		}
	};

	public ref class Parameter
	{
	public:
		SolverType SolverType;
		double Eps;
		double C;
		int NrThread;
		int NrWeight;
		array<int>^ WeightLabel;
		array<double>^ Weight;
		double p;
		array<double>^ InitSol;

		Parameter()
		{
			SolverType = SolverType::L2R_L2LOSS_SVC;
			Eps = 0.1;
			C = 1;
			NrThread = 1;
			NrWeight = 0;
			p = 0.1;
		}

		Parameter(LibLinear::SolverType solverType)
		{
			SolverType = solverType;
			Eps = 0.1;
			C = 1;
			NrThread = 1;
			NrWeight = 0;
			p = 0.1;
		}

	internal:

		Parameter(const parameter& native)
		{
			SolverType = (LibLinear::SolverType)native.solver_type;

			Eps = native.eps;

			C = native.C;

			NrThread = native.nr_thread;

			NrWeight = native.nr_weight;

			WeightLabel = gcnew array<int>(native.nr_weight);
			for (auto i = 0; i < native.nr_weight; i++) WeightLabel[i] = native.weight_label[i];

			Weight = gcnew array<double>(native.nr_weight);
			for (auto i = 0; i < native.nr_weight; i++) Weight[i] = native.weight[i];

			p = native.p;

			InitSol = gcnew array<double>(native.nr_weight);
			for (auto i = 0; i < native.nr_weight; i++) InitSol[i] = native.init_sol[i];
		}
	};

	public value struct Model
	{
		/// <summary>parameter</summary>
		Parameter^ Param;

		/// <summary>number of classes</summary>
		int NrClass;

		/// <summary>number of features</summary>
		int NrFeature;

		/// <summary>w</summary>
		array<double>^ W;

		/// <summary>label of each class (label[k])</summary>
		array<int>^ Label;
		
		/// <summary>bias</summary>
		double Bias;

		/* for classification only */

		/*/// <summary>
		/// 1 if svm_model is created by svm_load_model
		/// 0 if svm_model is created by svm_train
		/// </summary>
		int FreeSv;*/

		/// <summary>
		/// Gets solver type of this model.
		/// </summary>
		property SolverType Type {
			SolverType get() {
				return Param->SolverType;
			}
		}

	internal:

		Model(const model* native)
		{
			Param = gcnew Parameter(native->param);

			NrClass = native->nr_class;

			NrFeature = native->nr_feature;

			W = gcnew array<double>(native->nr_feature);
			Marshal::Copy((IntPtr)native->w, W, 0, native->nr_feature);

			Label = gcnew array<int>(native->nr_class);
			Marshal::Copy((IntPtr)native->label, Label, 0, native->nr_class);

			Bias = native->bias;
		}
	};

	public ref class Linear abstract sealed
	{
	public:

		static Model Train(String^ inputFileName)
		{
			auto nativeInputFileName = (char*)Marshal::StringToHGlobalAnsi(inputFileName).ToPointer();
			char* argv[] = {
				"", nativeInputFileName
			};
			auto model = liblinear_train::main(2, argv);
			auto m = Model(model);
			return m;
		}

		/// <summary>
		/// </summary>
		static Model Train(Problem problem, Parameter^ parameter)
		{
			::problem arg_problem;
			::parameter arg_parameter;

			try
			{
				// (1) convert managed Problem to native problem
				arg_problem = Convert(problem);

				// (2) convert managed Parameter to native parameter
				arg_parameter = Convert(parameter);

				// (3) call actual function
				auto r = train(&arg_problem, &arg_parameter);

				// debug code
				//svm_save_model("C:/Data/test_r1.txt", r);
				//svm_save_model("C:/Data/test_r2.txt", &Convert(Model(r)));

				// (4) convert resulting native model to managed Model
				auto result = Model(r);
				FreeNativeModel(r, false);
				return result;
			}
			finally
			{
				FreeProblem(&arg_problem);
				destroy_param(&arg_parameter);
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
		static array<double>^ CrossValidation(Problem problem, Parameter^ parameter, int nrFold, array<double>^ target)
		{
			::problem arg_problem;
			::parameter arg_parameter;
			double* native_target = NULL;

			try
			{
				arg_problem = Convert(problem);
				arg_parameter = Convert(parameter);

				auto l = problem.Count;
				auto target = new double[l];
				::cross_validation(&arg_problem, &arg_parameter, nrFold, target);

				auto result = gcnew array<double>(l);
				for (auto i = 0; i < l; i++) result[i] = target[i];
				return result;
			}
			finally
			{
				FreeProblem(&arg_problem);
				if (native_target != NULL) free(native_target);
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
			::feature_node* nativeNodes = NULL;
			::model nativeModel;
			try
			{
				nativeNodes = Convert(x);
				nativeModel = Convert(model);
				return ::predict(&nativeModel, nativeNodes);
			}
			finally
			{
				if (nativeNodes) delete[] nativeNodes;
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
			::feature_node* nativeNodes = NULL;
			::model nativeModel;
			try
			{
				nativeNodes = Convert(x);
				nativeModel = Convert(model);
				pin_ptr<double> pDecValues = &decValues[0];
				return ::predict_values(&nativeModel, nativeNodes, pDecValues);
			}
			finally
			{
				if (nativeNodes) delete[] nativeNodes;
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
			::feature_node* nativeNodes = NULL;
			::model nativeModel;
			try
			{
				nativeNodes = Convert(x);
				nativeModel = Convert(model);
				pin_ptr<double> pProbEstimates = &probEstimates[0];
				return ::predict_probability(&nativeModel, nativeNodes, pProbEstimates);
			}
			finally
			{
				if (nativeNodes) delete[] nativeNodes;
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
			auto err = ::save_model(nativeFileName, &native_model);

			if (err != 0) throw gcnew Exception("SaveModel failed with error code " + err + ".");
		}


		/// <summary>
		/// This function loads a model from file.
		/// </summary>
		static Model LoadModel(String^ fileName)
		{
			auto filenameNative = (char*)Marshal::StringToHGlobalAnsi(fileName).ToPointer();
			auto nativeModel = ::load_model(filenameNative);
			return Model(nativeModel);
		}




		/// <summary>
		/// This function checks whether the parameters are within the feasible
		/// range of the problem. This function should be called before 
		/// Svm.Train() and Svm.CrossValidation(). It returns null if the
		/// parameters are feasible, otherwise an error message is returned.
		/// </summary>
		static String^ CheckParameter(Problem problem, Parameter^ parameter)
		{
			::problem arg_problem;
			::parameter arg_parameter;

			try
			{
				arg_problem = Convert(problem);
				arg_parameter = Convert(parameter);
				auto r = ::check_parameter(&arg_problem, &arg_parameter);
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
			::model nativeModel;
			try
			{
				nativeModel = Convert(model);
				auto r = ::check_probability_model(&nativeModel);
				return r == 1;
			}
			finally
			{
				FreeNativeModel(&nativeModel, true);
			}

		}

	private:

		static ::problem Convert(Problem problem)
		{
			::problem result;

			result.l = problem.y->Length;
			result.n = problem.x[0]->Length;
			pin_ptr<double> yPinned = &problem.y[0];

			result.y = new double[result.l];
			memcpy(result.y, yPinned, result.l * sizeof(double));

			result.x = new ::feature_node*[result.l];
			for (int i = 0; i < result.l; i++) result.x[i] = Convert(problem.x[i]);

			result.bias = problem.Bias;

			return result;
		}

		static ::feature_node Convert(const Node node)
		{
			::feature_node x;
			x.index = node.Index;
			x.value = node.Value;
			return x;
		}

		static ::feature_node* Convert(array<Node>^ x)
		{
			auto n = x->Length;
			auto p = new ::feature_node[n + 1];
			pin_ptr<Node> pinned = &x[0];
			memcpy(p, pinned, n * sizeof(::feature_node));
			p[n].index = -1; p[n].value = 0.0; // add delimiter
			return p;
		}

		static ::parameter Convert(Parameter^ p)
		{
			::parameter result;
			result.solver_type = (int)p->SolverType;
			result.eps = p->Eps;
			result.C = p->C;
			result.nr_thread = p->NrThread;
			result.nr_weight = p->Weight ? p->Weight->Length : 0;
			if (p->WeightLabel && p->WeightLabel->Length)
			{
				auto count = p->WeightLabel->Length;

				result.weight_label = new int[count];
				pin_ptr<int> pinnedWeightLabel = &p->WeightLabel[0];
				memcpy(result.weight_label, pinnedWeightLabel, count * sizeof(int));

				result.weight = new double[count];
				pin_ptr<double> pinnedWeight = &p->Weight[0];
				memcpy(result.weight, pinnedWeight, count * sizeof(double));
			}
			else
			{
				result.weight_label = NULL;
				result.weight = NULL;
			}
			result.p = p->p;
			
			if (p->InitSol && p->InitSol->Length)
			{
				auto count = p->InitSol->Length;

				result.init_sol = new double[count];
				pin_ptr<double> pinnedInitSol = &p->InitSol[0];
				memcpy(result.init_sol, pinnedInitSol, count * sizeof(double));
			}
			else
			{
				result.init_sol = NULL;
			}

			return result;
		}

		static ::model Convert(Model model)
		{
			::model native;

			native.param = Convert(model.Param);
			native.nr_class = model.NrClass;
			native.nr_feature = model.NrFeature;
			COPY_MANAGED_ARRAY_TO_NATIVE(double, model.NrFeature, w, W)
			COPY_MANAGED_ARRAY_TO_NATIVE(int, model.NrClass, label, Label)
			native.bias = model.Bias;

			return native;
		}

		static void FreeProblem(::problem* problem)
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

		static void FreeNativeModel(::model* native, bool deleteInnerSVs)
		{
#define Clean(param) { if (native->param) delete[] native->param; native->param = NULL; }

			Clean(param.weight);
			Clean(param.weight_label);
			Clean(param.init_sol);
			Clean(label);
			Clean(w);
		}
	};
}
