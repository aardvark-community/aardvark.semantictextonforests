#include "liblinear.clr.h"

static void print_null(const char *) {}

namespace LibLinear
{
	void Linear::DisablePrint()
	{
		::set_print_string_function(print_null);
	}
}