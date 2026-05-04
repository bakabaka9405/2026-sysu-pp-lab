local function add_lab3_target(name, src)
	target(name, function()
		set_kind("binary")
		set_toolchains("gcc")
		add_includedirs("src")
		add_files(src)
		add_cxxflags("-pthread", {force = true})
		add_ldflags("-pthread", {force = true})
		if is_mode("release") then
			set_optimize("aggressive")
		end
	end)
end

target("lab3.gen", function()
	set_kind("binary")
	set_toolchains("gcc")
	add_files("src/gen.cpp")
	if is_mode("release") then
		set_optimize("aggressive")
	end
end)

add_lab3_target("lab3.task1", "src/task1.cpp")
add_lab3_target("lab3.task2", "src/task2.cpp")
