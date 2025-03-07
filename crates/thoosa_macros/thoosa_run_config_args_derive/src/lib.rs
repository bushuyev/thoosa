use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, Attribute, DeriveInput, Field, Lit, Path, Variant};
use thoosa_macros_utils::*;



#[proc_macro_derive(ThoosaRunConfig, attributes(run_config, run_config_source))]
pub fn thoosa_args_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let target_struct_ident = &ast.ident;

    eprintln!("thoosa_args_derive: target_struct_ident={target_struct_ident:?}");

    let range_or_value_struct_ident = Ident::new(
        &format!("{}RangeOrValue", target_struct_ident),
        target_struct_ident.span()
    );
    let fields_enum_ident = Ident::new(
        &format!("{}Fields", target_struct_ident),
        target_struct_ident.span()
    );


    let (range_or_value_fields, enum_fields, add_range_or_value_calls, range_or_value_type) = match &ast.data {
        syn::Data::Struct(data_struct) => {

            match &data_struct.fields {
                syn::Fields::Named(named_fields) => {

                    let (range_or_value_fields, enum_fields, add_range_or_value_calls) = named_fields.named.iter().fold((vec![], vec![], vec![]), |mut a, field| {
                        let field_ident = field.ident.as_ref().unwrap();
                        let camel_field_ident = Ident::new(&thoosa_shared::snake_to_camel(&field_ident.to_string()), fields_enum_ident.span());
                        let field_type = field.ty.clone();

                        eprintln!("field_type={}", field_type.to_token_stream());

                        if field.attrs.iter().any(|attr| attr.path().is_ident("run_config")) {
                            if let Some(inner_ty) = get_option_inner_ty(&field_type) {
                                if is_primitive(&inner_ty) {
                                    a.0.push(quote! {
                                        #field_ident: Option<RangeOrValue<#inner_ty>>,
                                    });
                                    a.1.push(quote! {
                                        #camel_field_ident,
                                    });
                                    a.2.push(quote! {
                                        if let Some(v) = self.#field_ident.as_ref(){
                                            thoosa_shared::add_range_or_value::<#inner_ty, T>(append(#fields_enum_ident::#camel_field_ident), v, const_params, grid_params, opt_params);
                                        }
                                    });

                                } else {
                                    //TODO
                                }
                            } else {
                                if is_primitive(&field_type) {
                                    a.0.push(quote! {
                                        #field_ident: RangeOrValue<#field_type>,
                                    });
                                        a.1.push(quote! {
                                        #camel_field_ident,
                                    });
                                        a.2.push(quote! {
                                        thoosa_shared::add_range_or_value::<#field_type, T>(append(#fields_enum_ident::#camel_field_ident), &self.#field_ident, const_params, grid_params, opt_params);
                                    });

                                } else {
                                    let field_type_range_or_value = Ident::new(&format!("{}RangeOrValue", field_type.to_token_stream().to_string()), camel_field_ident.span());

                                    a.0.push(quote! {
                                        #field_ident: #field_type_range_or_value,
                                    });

                                    let field_type_fields = Ident::new(&format!("{}Fields", field_type.to_token_stream().to_string()), camel_field_ident.span());
                                    a.1.push(quote! {
                                        #camel_field_ident(#field_type_fields),
                                    });
                                    a.2.push(quote! {
                                        self.#field_ident.fill_run_configs(|v| append(#fields_enum_ident::#camel_field_ident(v)), const_params, grid_params, opt_params);
                                    });
                                }

                            }

                        } else {
                            a.0.push(quote! {});
                            a.1.push(quote! {});
                            a.2.push(quote! {});
                        }

                        a
                    });

                    (
                        quote! { #(# range_or_value_fields)* },
                        quote! { #(# enum_fields)* },
                        quote! { #(# add_range_or_value_calls)* },
                        quote! {struct},
                    )
                },
                _ => {
                    println!("unnamed fields");
                    (
                        quote! {},
                        quote! {},
                        quote! {},
                        quote! {},
                    )
                }
            }
        },
        syn::Data::Enum(data_enum) => {
            let (range_or_value_fields, enum_fields, add_range_or_value_calls) = data_enum.variants.iter().fold((vec![], vec![], vec![]), |mut a, variant| {
                let variant_name = &variant.ident;

                if let syn::Fields::Unnamed(fields) = &variant.fields {
                    if fields.unnamed.len() == 1 {
                        let ty = &fields.unnamed.first().expect("no config type").ty;

                        let field_type_range_or_value = Ident::new(&format!("{}RangeOrValue", ty.to_token_stream().to_string()), variant.ident.span());

                        a.0.push(quote! {
                            #variant_name(#field_type_range_or_value),

                        });

                        let field_type_fields = Ident::new(&format!("{}Fields", ty.to_token_stream().to_string()), variant.ident.span());

                        a.1.push(quote! {
                            #variant_name(#field_type_fields),
                        });

                        a.2.push(quote! {
                            #range_or_value_struct_ident::#variant_name(v) => {
                                v.fill_run_configs(|v|append(#fields_enum_ident::#variant_name(v)), const_params, grid_params, opt_params);
                            }


                        });
                    }
                }

                a
            });

            (
                quote! { #(# range_or_value_fields)* },
                quote! { #(# enum_fields)* },
                quote! {  match self {#(# add_range_or_value_calls)* }},
                quote! {enum},
            )
        }
        _ => {
            (
                quote! {},
                quote! {},
                quote! {},
                quote! {}
            )
        }
    };

    let sources_fns = if ast.attrs.iter().any(|attr| {
        attr.path().is_ident("run_config_source")
    }) {
        quote! {
             fn opt_sources(&self) -> OptRunConfigSource<Vec<NamedVal<#fields_enum_ident>>, Vec<NamedRange<#fields_enum_ident>>, Vec<Hyperparameter<#fields_enum_ident>>, #fields_enum_ident> {
                let mut const_params = Vec::new();
                let mut grid_params = Vec::new();
                let mut opt_params = Vec::new();

                self.fill_run_configs(|v|v, &mut const_params, &mut grid_params, &mut opt_params);

                OptRunConfigSource::new(
                    GridRunConfigSource::new(
                        ConstConfigSource::new(const_params),
                        grid_params,
                    ),
                    opt_params,
                )
            }

            fn grid_sources(&self) -> GridRunConfigSource<Vec<NamedVal<#fields_enum_ident>>, Vec<NamedRange<#fields_enum_ident>>> {
                let mut const_params = Vec::new();
                let mut grid_params = Vec::new();
                let mut opt_params = Vec::new();

                self.fill_run_configs(|v|v, &mut const_params, &mut grid_params, &mut opt_params);
                if !opt_params.is_empty() {
                    panic!("opt_params in grid");
                }

                GridRunConfigSource::new(
                    ConstConfigSource::new(const_params),
                    grid_params,
                )
            }
        }
    } else {
        quote! {}
    };


    let gen = quote! {

        #[derive(Debug, serde::Deserialize, serde::Serialize)]
        #range_or_value_type #range_or_value_struct_ident {
            #range_or_value_fields

        }

        #[derive(Debug, PartialEq, Clone)]
        enum #fields_enum_ident {
            #enum_fields
        }

        impl Display for #fields_enum_ident {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                match self {
                    _=>{
                        write!(f, "todo")
                    }
                }
            }
        }

        impl FieldName for #fields_enum_ident {}

        impl #range_or_value_struct_ident {
            pub(crate) fn fill_run_configs<T:thoosa_shared::hyperparameter::FieldName, F>(&self, append: F, const_params: &mut Vec<thoosa_shared::val::NamedVal<T>>, grid_params: &mut Vec<thoosa_shared::val::NamedRange<T>>, opt_params: &mut Vec<thoosa_shared::hyperparameter::Hyperparameter<T>>)
                where F:Fn(#fields_enum_ident)->T{

                #add_range_or_value_calls
            }

            #sources_fns
        }
    };

    gen.into()
}