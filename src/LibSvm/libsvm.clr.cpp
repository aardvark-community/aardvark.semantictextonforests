#include "libsvm.clr.h"

static void print_null(const char *) {}

namespace LibSvm
{
	void Svm::DisablePrint()
	{
		::svm_set_print_string_function(print_null);
	}
}
