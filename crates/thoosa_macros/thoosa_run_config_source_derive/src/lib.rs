use proc_macro::TokenStream;
use quote::quote;

#[proc_macro_derive(RunConfigSource, attributes(run_config))]
pub fn run_config_source_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let target_struct = &ast.ident;

    let in_fields = if let syn::Data::Struct(syn::DataStruct { fields: syn::Fields::Named(syn::FieldsNamed { ref named, .. }), .. }) = ast.data {
        named
    } else {
        panic!("no fields found");
    };

    let in_run_config_keys = in_fields.iter().filter(|f| {
        f.attrs.iter().any(|a|a.path().is_ident("run_config"))
    }).map(|f|{
        f.ident.as_ref().map(|field_ident| {
            let ident_lit = field_ident.to_string();
            let new_ident = syn::Ident::new(&ident_lit, field_ident.span());

            quote! {
                 match self.#new_ident {
                        RangeOrValue::Range { range, step: step_op } => {

                            match step_op {
                                None => {
                                    opt_params.push(
                                        (#ident_lit, range).into()
                                    );
                                }
                                Some((step, mul)) => {
                                    grid_params.push(
                                        NamedRange::new(
                                            #ident_lit,
                                            range.start.into()..range.end.into(),
                                            (step.into(), mul)
                                        )
                                    );
                                }
                            }
                        }

                        RangeOrValue::Value(v) => {
                            const_params.push(
                                NamedVal::from((#ident_lit, v.into()))
                            );
                        }
                    }

            }
        })
    });
    // eprintln!("in_run_config_keys={in_run_config_keys:?}");



    let gen = quote! {
        use log::debug;

        impl RunConfigSource for #target_struct  {
            type CC = VecConfig<NamedVal>;
            type GC = VecConfig<NamedRange>;
            type OC = VecConfig<Hyperparameter>;

            fn to_run_configs(self) -> RunConfigs<Self::CC, Self::GC, Self::OC> {


                let (mut const_params, mut grid_params, mut opt_params):(Vec<NamedVal>, Vec<NamedRange>, Vec<Hyperparameter>) = (vec![], vec![], vec![]);


                #(#in_run_config_keys)*;

                RunConfigs::new(
                    const_params.into(),
                    if grid_params.is_empty() {None} else {Some(grid_params.into())},
                    if opt_params.is_empty() {None} else {Some(opt_params.into())},
                )
            }
        }
    };
    gen.into()
}


#[cfg(test)]
mod tests {
    use super::*;

}
