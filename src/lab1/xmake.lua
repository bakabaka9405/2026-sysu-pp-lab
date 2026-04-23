add_requires("msmpi")

target("lab1.gen", function()
    set_kind("binary")
    add_files("src/gen.cpp")
end)

target("lab1", function()
    set_kind("binary")
    add_files("src/main.cpp")
    add_packages("msmpi")
    if is_mode("release") then
        set_optimize("aggressive")
    end
end)