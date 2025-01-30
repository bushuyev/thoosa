use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{Field, Type};
use quote::format_ident;
use syn::spanned::Spanned;
use thoosa_run_config_source::*;



fn type_to_string(ty: &syn::Type) -> String {
    quote! { #ty }.to_string()
    // "".to_string()
}

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

    let mut field_metas = Vec::new(); // `Vec<quote! {...}>` or some struct to store data
    let mut sub_types = Vec::new();   // For each flattened field, store the type


    fn process_field(field: &Field, run_config_fields: &mut Vec<proc_macro2::TokenStream>, field_assignments: &mut Vec<proc_macro2::TokenStream>) {
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

            if let Some(attr) = field.attrs.iter().find(|attr| attr.to_token_stream().into_iter().any(|v|v.to_string().contains("flatten"))) {
                eprintln!("attr={attr:?} field.ty={:?}", field.ty);

                // if let Type::Path(type_path) = &field.ty {
                //     // Use reflection to process the fields of the flattened struct
                //     let flattened_struct = &type_path.path.segments.last().unwrap().ident;
                //     println!("Flattening struct: {}", flattened_struct);
                //     if let Some(flattened_ast) = get_struct_ast(flattened_struct) {
                //         if let Data::Struct(data_struct) = flattened_ast.data {
                //             if let Fields::Named(fields_named) = data_struct.fields {
                //                 for inner_field in fields_named.named {
                //                     process_field(&inner_field);
                //                 }
                //             }
                //         }
                //     }
                // }
            }

        }
    }

    for field in in_fields {
        process_field(field, &mut run_config_fields, &mut field_assignments);

        let field_ident = field.ident.as_ref().unwrap();
        let field_name_str = field_ident.to_string();



        if let Some(attr) = field.attrs.iter().find(|attr| attr.to_token_stream().into_iter().any(|v|v.to_string().contains("flatten"))) {
            // We'll recursively gather fields from that type by calling <ThatType as FlattenAllFields>::all_fields()
            // Also keep track of the sub-type for building a "flattened struct"
            let field_ty = &field.ty;
            sub_types.push(field_ty.clone());
        } else {
            // A normal field: store its name and type
            let ty_str = type_to_string(&field.ty);
            field_metas.push(quote! {
                    FieldMeta {
                        name: #field_name_str.to_string(),
                        ty_str: #ty_str.to_string(),
                    }
                });
        }


        // eprintln!("field={field:?}");
        // for attr in &field.attrs {
        //     eprintln!("attr={attr:?}");
        //     attr.to_token_stream().into_iter().find(|t|t.)
        // }

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

    let mut sub_flatten_calls = Vec::new();
    for ty in sub_types.iter() {
        sub_flatten_calls.push(quote! {
            .into_iter().chain(<#ty as FlattenAllFields>::all_fields().into_iter().cloned())
        });
    }

    let fields_array_ident = format_ident!("_IMPL_FLATTENED_{}_FIELDS", target_struct);

    // We'll define a const array of `FieldMeta` to hold the fields.
    let expanded_trait_impl = quote! {
        impl FlattenAllFields for #target_struct {
            fn all_fields() -> &'static [FieldMeta] {
                // A static "Lazy" storing our final Vec<FieldMeta>
                static #fields_array_ident: once_cell::sync::Lazy<Vec<FieldMeta>> = once_cell::sync::Lazy::new(|| {
                    // Start from our base fields as an iterator
                    let base_iter = vec![
                        #(#field_metas),*
                    ].into_iter()

                    // Then chain sub-types' fields (if any):
                    #(#sub_flatten_calls)*

                    // Finally collect all into a Vec
                    .collect::<Vec<FieldMeta>>();

                    base_iter
                });

                // Return a &'static [FieldMeta]
                #fields_array_ident.as_slice()
            }
        }
    };

    let flattened_struct_ident = format_ident!("Flattened{}", target_struct);

    let mut flatten_struct_fields = Vec::new();
    for (i, meta) in field_metas.iter().enumerate() {

        let field_i_str = syn::Ident::new(format!("field_{}", i).as_str(), meta.span());
        flatten_struct_fields.push(quote! {
            pub #field_i_str: String
        });
    }

    let expanded_struct = quote! {
        #[allow(non_snake_case)]
        pub struct #flattened_struct_ident {
            #(#flatten_struct_fields),*
        }
    };


    let gen = quote! {


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



        #expanded_trait_impl

        #expanded_struct
    };
    gen.into()
}


#[cfg(test)]
mod tests {}
