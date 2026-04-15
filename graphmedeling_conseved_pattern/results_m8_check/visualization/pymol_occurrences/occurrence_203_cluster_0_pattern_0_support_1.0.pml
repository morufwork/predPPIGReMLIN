load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6m0j.ent", occ_203_c0_p0_s1.0
hide everything, occ_203_c0_p0_s1.0
show cartoon, occ_203_c0_p0_s1.0 and chain A+E
color palegreen, occ_203_c0_p0_s1.0 and chain A
color lightblue, occ_203_c0_p0_s1.0 and chain E
select hotspot_source, occ_203_c0_p0_s1.0 and ((chain A and resi 79))
select hotspot_target, occ_203_c0_p0_s1.0 and ((chain E and resi 486))
select hotspot_all, occ_203_c0_p0_s1.0 and ((chain A and resi 79) or (chain E and resi 486))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_203_c0_p0_s1.0 and chain A+E
set_name hotspot_all, hotspot_occurrence_203
set_name hotspot_source, hotspot_source_203
set_name hotspot_target, hotspot_target_203
bg_color white
# patternId=0 support=1.0 graphId=20
