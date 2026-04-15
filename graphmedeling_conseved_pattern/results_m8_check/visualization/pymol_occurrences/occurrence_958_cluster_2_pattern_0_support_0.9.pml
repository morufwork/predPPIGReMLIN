load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xoc.ent", occ_958_c2_p0_s0.9
hide everything, occ_958_c2_p0_s0.9
show cartoon, occ_958_c2_p0_s0.9 and chain D+A
color palegreen, occ_958_c2_p0_s0.9 and chain D
color lightblue, occ_958_c2_p0_s0.9 and chain A
select hotspot_source, occ_958_c2_p0_s0.9 and ((chain D and resi 38))
select hotspot_target, occ_958_c2_p0_s0.9 and ((chain A and resi 498))
select hotspot_all, occ_958_c2_p0_s0.9 and ((chain A and resi 498) or (chain D and resi 38))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_958_c2_p0_s0.9 and chain D+A
set_name hotspot_all, hotspot_occurrence_958
set_name hotspot_source, hotspot_source_958
set_name hotspot_target, hotspot_target_958
bg_color white
# patternId=0 support=0.9 graphId=357
