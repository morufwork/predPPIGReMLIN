load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7v84.ent", occ_2302_c4_p0_s0.9
hide everything, occ_2302_c4_p0_s0.9
show cartoon, occ_2302_c4_p0_s0.9 and chain A+F
color palegreen, occ_2302_c4_p0_s0.9 and chain A
color lightblue, occ_2302_c4_p0_s0.9 and chain F
select hotspot_source, occ_2302_c4_p0_s0.9 and ((chain A and resi 486))
select hotspot_target, occ_2302_c4_p0_s0.9 and ((chain F and resi 82))
select hotspot_all, occ_2302_c4_p0_s0.9 and ((chain A and resi 486) or (chain F and resi 82))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2302_c4_p0_s0.9 and chain A+F
set_name hotspot_all, hotspot_occurrence_2302
set_name hotspot_source, hotspot_source_2302
set_name hotspot_target, hotspot_target_2302
bg_color white
# patternId=0 support=0.9 graphId=228
