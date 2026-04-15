load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7efp.ent", occ_211_c0_p0_s1.0
hide everything, occ_211_c0_p0_s1.0
show cartoon, occ_211_c0_p0_s1.0 and chain A+B
color palegreen, occ_211_c0_p0_s1.0 and chain A
color lightblue, occ_211_c0_p0_s1.0 and chain B
select hotspot_source, occ_211_c0_p0_s1.0 and ((chain A and resi 79))
select hotspot_target, occ_211_c0_p0_s1.0 and ((chain B and resi 486))
select hotspot_all, occ_211_c0_p0_s1.0 and ((chain A and resi 79) or (chain B and resi 486))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_211_c0_p0_s1.0 and chain A+B
set_name hotspot_all, hotspot_occurrence_211
set_name hotspot_source, hotspot_source_211
set_name hotspot_target, hotspot_target_211
bg_color white
# patternId=0 support=1.0 graphId=63
