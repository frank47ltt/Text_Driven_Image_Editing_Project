####################################################################
# Editing with CD for different hyperparameters
####################################################################

for crs_srs in "0.3 0.6"
do
    echo $crs_srs | read crs srs
    echo "CD: cross_replace_steps: $crs, self_replace_steps $srs"
    CUDA_VISIBLE_DEVICES=0 python3.10 sd1.5/edit.py --model_id_DM runwayml/stable-diffusion-v1-5         \
                                                        --reverse_checkpoint ../iCD-SD15_4steps_2/iCD-SD15-reverse_249_499_699_999.safetensors           \
                                                           --forward_checkpoint ../iCD-SD15_4steps_2/iCD-SD15-forward_19_249_499_699.safetensors            \
                                                           --teacher_checkpoint ../iCD-SD15_4steps_2/sd15_cfg_distill.pt             \
                                                           --num_reverse_cons_steps 4                   \
                                                           --reverse_timesteps 249 499 699 999          \
                                                           --num_forward_cons_steps 4                   \
                                                           --forward_timesteps 19 249 499 699           \
                                                           --path_to_prompts benchmarks/instructions/editing_pie_bench_all.csv         \
                                                           --path_to_images benchmarks/images/pie_bench_all_images             \
                                                           --guidance_scale 19                          \
                                                           --dynamic_guidance True                      \
                                                           --tau1 0.8                                   \
                                                           --tau2 0.8                                   \
                                                           --cross_replace_steps $crs                     \
                                                           --self_replace_steps $srs                      \
                                                           --amplify_factor 4                           \
                                                           --max_forward_timestep_idx 49                \
                                                           --start_timestep 19                          \
                                                           --nti_guidance_scale 8.0                     \
                                                           --use_npi False                              \
                                                           --use_nti False                              \
                                                           --use_cons_inversion True                    \
                                                           --use_cons_editing True                      \
                                                           --lora_rank 64                               \
                                                           --w_embed_dim 512                            \
                                                           --num_ddim_steps 50                          \
                                                           --device cpu                                \
                                                           --saving_dir results_editing_cons/results_${crs}_${srs} \
                                                           --dtype fp32                                 \
                                                           --is_replacement False                       \
                                                           --seed 30
done
