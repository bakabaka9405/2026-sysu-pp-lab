target("lab0.gen",function()
	set_toolchains("gcc")
	add_files("src/gen.cpp")
end)

target("lab0.python", function()
	set_kind("phony")
	on_run(function (target)
		local scriptdir = target:scriptdir()
        os.execv("python", {path.join(scriptdir, "src/baseline.py")})
    end)
end)

target("lab0.baseline", function()
	set_toolchains("gcc")
	add_files("src/baseline.cpp")
	set_rules("mode.debug")
end)

target("lab0.switch-loop-order", function()
	set_toolchains("gcc")
	add_files("src/switch_loop_order.cpp")
	set_rules("mode.debug")
end)

target("lab0.flatten_loops", function()
	set_toolchains("gcc")
	add_files("src/flatten_loop.cpp")
	set_rules("mode.debug")
end)

target("lab0.optimize", function()
	set_toolchains("gcc")
	add_files("src/flatten_loop.cpp")
	set_rules("mode.release")
end)

local ONEAPI_DIR = "C:/Program Files (x86)/Intel/oneAPI"
local MKL_DIR = path.join(ONEAPI_DIR, "mkl/latest")

target("lab0.mkl", function()
	set_toolchains("icx")
	add_files("src/mkl.cpp")

	add_sysincludedirs(path.join(ONEAPI_DIR, "compiler/latest/include"))
	add_sysincludedirs(path.join(MKL_DIR, "include"))
	add_linkdirs(path.join(MKL_DIR, "lib"))
	set_runtimes("MD")
	add_syslinks(
		"sycl",
		"OpenCL",
		"mkl_sycl_dll",
		"mkl_intel_lp64_dll", 
		"mkl_sequential_dll", 
		"mkl_core_dll"
	)
end)