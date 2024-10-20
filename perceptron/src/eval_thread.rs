use ia_engine::trainer::{default_param_translator, CriticalityCue, Trainer};

use crate::neuronal_network::neuronal_network;

pub fn eval_thread(pixel_input: &[f32; 28 * 28]) -> [f32; 10] {
    let mut trainer = Trainer::new_hybrid(
        CriticalityCue::<{ 63610 / 4 }>(),
        neuronal_network::<{ 28 * 28 }, 10, 63610, _>,
        neuronal_network::<{ 28 * 28 }, _, _, _>,
        default_param_translator,
        // param_translator_with_bounds::<_, 4, -4>,
        vec![28 * 28, 80, 10],
    );

    trainer.load("model.bin").unwrap();

    trainer.eval(pixel_input)
}
