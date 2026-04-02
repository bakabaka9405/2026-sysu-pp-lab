add_rules("mode.debug", "mode.release")
set_languages("c++20")
set_rundir("$(projectdir)")

includes("**/xmake.lua")