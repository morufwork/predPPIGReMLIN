load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7dhx.ent", occ_71_c0_p0_s0.8
hide everything, occ_71_c0_p0_s0.8
show cartoon, occ_71_c0_p0_s0.8 and chain A+B
color palegreen, occ_71_c0_p0_s0.8 and chain A
color lightblue, occ_71_c0_p0_s0.8 and chain B
select hotspot_source, occ_71_c0_p0_s0.8 and ((chain A and resi 82))
select hotspot_target, occ_71_c0_p0_s0.8 and ((chain B and resi 486))
select hotspot_all, occ_71_c0_p0_s0.8 and ((chain A and resi 82) or (chain B and resi 486))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_71_c0_p0_s0.8 and chain A+B
set_name hotspot_all, hotspot_occurrence_71
set_name hotspot_source, hotspot_source_71
set_name hotspot_target, hotspot_target_71
bg_color white
# patternId=0 support=0.8 graphId=37
