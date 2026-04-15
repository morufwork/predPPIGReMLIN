load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7v84.ent", occ_2773_c5_p0_s0.8
hide everything, occ_2773_c5_p0_s0.8
show cartoon, occ_2773_c5_p0_s0.8 and chain A+F
color palegreen, occ_2773_c5_p0_s0.8 and chain A
color lightblue, occ_2773_c5_p0_s0.8 and chain F
select hotspot_source, occ_2773_c5_p0_s0.8 and ((chain A and resi 501) or (chain A and resi 505))
select hotspot_target, occ_2773_c5_p0_s0.8 and ((chain F and resi 353))
select hotspot_all, occ_2773_c5_p0_s0.8 and ((chain A and resi 501) or (chain A and resi 505) or (chain F and resi 353))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2773_c5_p0_s0.8 and chain A+F
set_name hotspot_all, hotspot_occurrence_2773
set_name hotspot_source, hotspot_source_2773
set_name hotspot_target, hotspot_target_2773
bg_color white
# patternId=0 support=0.8 graphId=232
