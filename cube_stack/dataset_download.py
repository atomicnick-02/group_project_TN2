from huggingface_hub import hf_hub_download
 
for variant in ['stack_d0.hdf5', 'stack_d1.hdf5']:
    hf_hub_download(
        repo_id='amandlek/mimicgen_datasets',
        filename=f'core/{variant}',
        repo_type='dataset',
        local_dir='./datasets')
    print(f'Downloaded {variant}')