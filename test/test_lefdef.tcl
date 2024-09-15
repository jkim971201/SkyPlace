read_lef "../test/ng45_tech.lef"
read_lef "../test/ng45_macro.lef"
read_def "../test/ng45_aes.def"

global_place -target_density 0.5
write_def write_test_aes.def

display
