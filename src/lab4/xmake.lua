local function add_lab4_target(name, src)
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

add_lab4_target("lab4.task1", "src/task1.cpp")
add_lab4_target("lab4.task2", "src/task2.cpp")
