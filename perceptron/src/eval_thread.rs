use ia_engine::trainer::{default_param_translator, CriticalityCue, Trainer};

use crate::neuronal_network::neuronal_network;

pub fn eval_thread(pixel_input: &[f32; 14 * 14]) -> [f32; 10] {
    let mut trainer = Trainer::new_hybrid(
        CriticalityCue::<{ 6740 / 20 }>(),
        neuronal_network::<{ 14 * 14 }, 10, 6740, _>,
        neuronal_network::<{ 14 * 14 }, _, _, _>,
        default_param_translator,
        // param_translator_with_bounds::<_, 4, -4>,
        vec![14 * 14, 30, 20, 10],
    );


    trainer.load("model.bin").unwrap();

    trainer.eval(pixel_input)
}
