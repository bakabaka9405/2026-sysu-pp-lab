add_requires("msmpi")

local function add_lab2_target(name, src)
	target(name, function()
		set_kind("binary")
		add_includedirs("src")
		add_files(src)
		add_packages("msmpi")
		if is_mode("release") then
			set_optimize("aggressive")
		end
	end)
end

add_lab2_target("lab2.task1", "src/task1.cpp")
add_lab2_target("lab2.task2", "src/task2.cpp")
add_lab2_target("lab2.task3", "src/task3.cpp")
