use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Field, Type};

#[proc_macro_derive(RunConfigSource, attributes(run_config))]
pub fn run_config_source_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let target_struct = &ast.ident;

    let target_struct_name = target_struct.to_string();
    let run_config_struct = syn::Ident::new(
        &if target_struct_name.contains("Args") {
            target_struct_name.replace("Args", "RunConfig")
        } else {
            format!("{target_struct_name}RunConfig")
        },
        target_struct.span(),
    );

    let in_fields = if let syn::Data::Struct(syn::DataStruct { fields: syn::Fields::Named(syn::FieldsNamed { ref named, .. }), .. }) = ast.data {
        named
    } else {
        panic!("no fields found");
    };

    let mut run_config_fields = Vec::new();
    let mut field_assignments = Vec::new();

    for field in in_fields {
        // Extract the field type
        if let Type::Path(type_path) = &field.ty {
            let last_segment = type_path.path.segments.last().unwrap();

            // Check if the type is `RangeOrValue`
            if last_segment.ident == "RangeOrValue" {
                // Extract the type parameter `T` from `RangeOrValue<T>`
                if let syn::PathArguments::AngleBracketed(angle_bracketed) = &last_segment.arguments {
                    if let Some(syn::GenericArgument::Type(ty)) = angle_bracketed.args.first() {
                        // Generate a new field with the extracted type
                        let field_ident = field.ident.as_ref().unwrap();
                        run_config_fields.push(quote! {
                            pub #field_ident: #ty
                        });

                        // Generate field assignment for the `From` implementation
                        let field_name = field_ident.to_string();
                        field_assignments.push(quote! {
                            #field_ident: values_hash
                                .get(#field_name)
                                .expect(#field_name)
                                .try_into()
                                .expect(&format!("Failed to parse {} value", #field_name))
                        });
                    }
                }
            }
        }
    }

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

        #[derive(Debug, Clone)]
        struct #run_config_struct  {
            #(#run_config_fields,)*
        }

        impl From<&[NamedVal]> for #run_config_struct {
            fn from(values: &[NamedVal]) -> Self {
                let values_hash = values.iter().fold(HashMap::new(), |mut acc, x| {
                    acc.insert(x.name(), x.val().clone());
                    acc
                });
                Self {
                    #(#field_assignments,)*
                }
            }
        }
    };
    gen.into()
}


#[cfg(test)]
mod tests {
    use super::*;

}
