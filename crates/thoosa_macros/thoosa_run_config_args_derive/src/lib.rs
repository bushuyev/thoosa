use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::{format_ident, quote, ToTokens};
use syn::{parse2, parse_macro_input, Attribute, DeriveInput, Field, Lit, Path, Variant};
use syn::spanned::Spanned;
use thoosa_macros_utils::*;

const RUN_CONFIG_ATTR: &'static str = "run_config";

#[proc_macro_derive(ThoosaRunConfig, attributes(run_config, run_config_opt, run_config_grid))]
pub fn thoosa_args_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let target_struct_ident = &ast.ident;
    let target_struct_name = &ast.ident.to_string();

    eprintln!("thoosa_args_derive: target_struct_ident={target_struct_ident:?}");

    let range_or_value_struct_ident = format_ident!("{}RangeOrValue", target_struct_ident);
    let fields_enum_ident = format_ident!("{}Fields", target_struct_ident);
    let builder_ident = format_ident!("{}Builder", target_struct_ident);
    let builder_enum_ident = format_ident!("{}BuilderEnum", target_struct_ident);

    struct TokenStreams {
        range_or_value_fields:Vec<proc_macro2::TokenStream>,
        range_or_value_add_calls:Vec<proc_macro2::TokenStream>,
        enum_fields:Vec<proc_macro2::TokenStream>,
        field_names:Vec<String>,

        builder_fields:Vec<proc_macro2::TokenStream>,
        builder_add_val_match_cases:Vec<proc_macro2::TokenStream>,
        builder_assignments:Vec<proc_macro2::TokenStream>,
        builder_enum_variants:Vec<proc_macro2::TokenStream>
    }

    impl TokenStreams {

        pub fn new() -> Self {
            Self {
                range_or_value_fields:vec![],
                range_or_value_add_calls:vec![],
                enum_fields:vec![],
                field_names:vec![],

                builder_fields:vec![],
                builder_add_val_match_cases:vec![],
                builder_assignments:vec![],
                builder_enum_variants:vec![]
            }
        }
    }


    let (
        range_or_value_fields,
        enum_fields,
        enum_fields_for_display,
        range_or_value_add_calls,
        range_or_value_type,
        //builder
        builder_fields,
        builder_add_val_match_cases,
        builder_assignments,
        builder_enum_variants,
    ) = match &ast.data {
        syn::Data::Struct(data_struct) => {

            match &data_struct.fields {
                syn::Fields::Named(named_fields) => {

                    let  token_streams = named_fields.named.iter().fold(TokenStreams::new(), |mut a, field| {
                        let field_ident = field.ident.as_ref().unwrap();
                        let camel_field_ident = Ident::new(&thoosa_shared::snake_to_camel(&field_ident.to_string()), fields_enum_ident.span());
                        let field_type = field.ty.clone();

                        eprintln!("field_type={}", field_type.to_token_stream());

                        if has_attr(field, RUN_CONFIG_ATTR) {


                            if let Some(inner_ty) = get_option_inner_ty(&field_type) {//is Option

                                if is_primitive(&inner_ty) {
                                    a.range_or_value_fields.push(quote! {
                                        #field_ident: Option<thoosa_shared::RangeOrValue<#inner_ty>>,
                                    });

                                    a.enum_fields.push(quote! {
                                        #camel_field_ident
                                    });
                                    a.field_names.push(field_ident.to_string());


                                    a.range_or_value_add_calls.push(quote! {
                                        if let Some(v) = self.#field_ident.as_ref(){
                                            thoosa_shared::add_range_or_value::<#inner_ty, T>(append(#fields_enum_ident::#camel_field_ident), v, const_params, grid_params, opt_params);
                                        }
                                    });

                                    a.builder_fields.push(quote! {
                                        #field_ident:Option<#inner_ty>,
                                    });

                                    a.builder_add_val_match_cases.push(quote! {
                                        #fields_enum_ident::#camel_field_ident => {
                                            self.#field_ident = Some(val.try_into().expect(&format!("#field_ident field value {val:?} is not valid")));
                                        }
                                    });

                                    a.builder_assignments.push(quote!{
                                        #field_ident: self.#field_ident,
                                    });

                                } else {
                                    let field_type_range_or_value = Ident::new(&format!("{}RangeOrValue", inner_ty.to_token_stream().to_string()), camel_field_ident.span());

                                    a.range_or_value_fields.push(quote! {
                                        #field_ident: Option<#field_type_range_or_value>,
                                    });

                                    let field_type_fields = Ident::new(&format!("{}Fields", inner_ty.to_token_stream().to_string()), camel_field_ident.span());
                                    a.enum_fields.push(quote! {
                                        #camel_field_ident(#field_type_fields)
                                    });
                                    a.field_names.push(field_ident.to_string());

                                    a.range_or_value_add_calls.push(quote! {
                                        if let Some(v) = self.#field_ident.as_ref(){
                                            v.fill_run_configs(|v| append(#fields_enum_ident::#camel_field_ident(v)), const_params, grid_params, opt_params);
                                        }
                                    });

                                    let field_type_builder = Ident::new(&format!("{}Builder", inner_ty.to_token_stream().to_string()), camel_field_ident.span());
                                    a.builder_fields.push(quote! {
                                        #field_ident:Option<#field_type_builder>,
                                    });

                                    a.builder_add_val_match_cases.push(quote! {
                                        #fields_enum_ident::#camel_field_ident(sub_path) => {
                                            if let Some(builder) = self.#field_ident.as_mut() {
                                                builder.add_val(sub_path, val);

                                            } else {

                                                let mut builder = #field_type_builder::default();
                                                builder.add_val(sub_path, val);
                                                self.#field_ident = Some(builder);
                                            }

                                        }
                                    });

                                    a.builder_assignments.push(quote!{
                                        #field_ident: self.#field_ident.as_ref().map(|b|b.build()),
                                    });
                                }
                            } else {
                                if is_primitive(&field_type) {
                                    a.range_or_value_fields.push(quote! {
                                        #field_ident: thoosa_shared::RangeOrValue<#field_type>,
                                    });
                                    a.enum_fields.push(quote! {
                                        #camel_field_ident
                                    });
                                    a.field_names.push(field_ident.to_string());

                                    a.range_or_value_add_calls.push(quote! {
                                        thoosa_shared::add_range_or_value::<#field_type, T>(append(#fields_enum_ident::#camel_field_ident), &self.#field_ident, const_params, grid_params, opt_params);
                                    });

                                    a.builder_fields.push(quote! {
                                        #field_ident:Option<#field_type>,
                                    });

                                    a.builder_add_val_match_cases.push(quote! {
                                        #fields_enum_ident::#camel_field_ident => {
                                            self.#field_ident = Some(val.try_into().expect(&format!("#field_ident field value {val:?} is not valid")));
                                        }
                                    });
                                    let expect_msg = format!("no {}", field_ident);

                                    a.builder_assignments.push(quote!{
                                        #field_ident: self.#field_ident.expect(#expect_msg),
                                    });

                                } else {
                                    let field_type_range_or_value = Ident::new(&format!("{}RangeOrValue", field_type.to_token_stream().to_string()), camel_field_ident.span());

                                    a.range_or_value_fields.push(quote! {
                                        #field_ident: #field_type_range_or_value,
                                    });

                                    let field_type_fields = Ident::new(&format!("{}Fields", field_type.to_token_stream().to_string()), camel_field_ident.span());
                                    a.enum_fields.push(quote! {
                                        #camel_field_ident(#field_type_fields)
                                    });
                                    a.field_names.push(field_ident.to_string());

                                    a.range_or_value_add_calls.push(quote! {
                                        self.#field_ident.fill_run_configs(|v| append(#fields_enum_ident::#camel_field_ident(v)), const_params, grid_params, opt_params);
                                    });

                                    let field_type_builder = Ident::new(&format!("{}Builder", field_type.to_token_stream().to_string()), camel_field_ident.span());
                                    a.builder_fields.push(quote! {
                                        #field_ident:#field_type_builder,
                                    });

                                    a.builder_add_val_match_cases.push(quote! {
                                        #fields_enum_ident::#camel_field_ident(sub_path) => {
                                            self.#field_ident.add_val(sub_path, val);
                                        }
                                    });

                                    a.builder_assignments.push(quote!{
                                        #field_ident: self.#field_ident.build(),
                                    });
                                }

                            }

                        } else {
                            a.range_or_value_fields.push(quote! {
                                #field_ident: #field_type,
                            });
                            a.enum_fields.push(quote! {
                                #camel_field_ident
                            });
                            a.field_names.push(field_ident.to_string());


                            if let Some(inner_ty) = get_option_inner_ty(&field_type) { //is Option
                                a.builder_fields.push(quote! {
                                    #field_ident:#field_type,
                                });

                                a.range_or_value_add_calls.push(quote! {

                                    if let Some(v) = self.#field_ident.as_ref(){
                                        const_params.push(thoosa_shared::val::NamedVal::from((append(#fields_enum_ident::#camel_field_ident), v.clone().into())));
                                    }

                                });

                                let field_type_str = field_type.to_token_stream().to_string();
                                let field_ident_str = field_ident.to_string();
                                a.builder_add_val_match_cases.push(quote! {
                                    #fields_enum_ident::#camel_field_ident => {
                                        let v:#inner_ty = val.try_into().expect(&format!("can't convert val {:?}:{} to {}", val, #field_ident_str, #field_type_str));
                                        self.#field_ident = Some(v);
                                    }
                                });

                                let expect_msg = format!("no {}", field_ident);

                                a.builder_assignments.push(quote!{
                                    #field_ident: self.#field_ident.clone(),
                                });

                            } else {
                                a.builder_fields.push(quote! {
                                    #field_ident:Option<#field_type>,
                                });

                                a.range_or_value_add_calls.push(quote! {
                                    const_params.push(thoosa_shared::val::NamedVal::from((append(#fields_enum_ident::#camel_field_ident), self.#field_ident.clone().into())));
                                });

                                let field_type_str = field_type.to_token_stream().to_string();
                                let field_ident_str = field_ident.to_string();
                                a.builder_add_val_match_cases.push(quote! {
                                    #fields_enum_ident::#camel_field_ident => {
                                        let v:#field_type = val.try_into().expect(&format!("can't convert val {:?}:{} to {}", val, #field_ident_str, #field_type_str));
                                        self.#field_ident = Some(v.clone());
                                    }
                                });

                                let expect_msg = format!("no {}", field_ident);

                                a.builder_assignments.push(quote!{
                                    #field_ident: self.#field_ident.clone().expect(#expect_msg),
                                });
                            }

                        }

                        a
                    });

                    let range_or_value_fields = token_streams.range_or_value_fields;
                    let enum_fields = token_streams.enum_fields;
                    let field_names = token_streams.field_names;
                    let builder_fields = token_streams.builder_fields;
                    let add_range_or_value_calls = token_streams.range_or_value_add_calls;
                    let add_val_match_cases = token_streams.builder_add_val_match_cases;
                    let builder_assignments = token_streams.builder_assignments;

                    (
                        quote! { #(# range_or_value_fields)* },
                        quote! { #(# enum_fields,)* },
                        quote! { #( Self::#enum_fields => { write!(f, #field_names)})* },
                        quote! { #(# add_range_or_value_calls)* },
                        quote! {struct},
                        quote! { #(# builder_fields)* },
                        quote! { match path { #(# add_val_match_cases)*} },
                        quote! {

                            #target_struct_ident {
                                #(# builder_assignments)*
                            }
                        },
                        quote! {},
                    )
                },
                _ => {
                    println!("unnamed fields");
                    (
                        quote! {},
                        quote! {},
                        quote! {},
                        quote! {},
                        quote! {},
                        quote! {},
                        quote! {},
                        quote! {},
                        quote! {}
                    )
                }
            }
        },
        syn::Data::Enum(data_enum) => {
            let builder_enum_field = Ident::new(&thoosa_shared::camel_to_snake(&builder_enum_ident.to_string()), data_enum.variants.span());

            let mut token_streams = data_enum.variants.iter().fold(TokenStreams::new(), |mut a, variant| {
                let variant_ident = &variant.ident;



                if let syn::Fields::Unnamed(fields) = &variant.fields {
                    if fields.unnamed.len() == 1 {
                        let ty = &fields.unnamed.first().expect("no config type").ty;

                        let ty_string = ty.to_token_stream().to_string();
                        let field_type_range_or_value = Ident::new(&format!("{}RangeOrValue", ty_string), variant.ident.span());

                        a.range_or_value_fields.push(quote! {
                            #variant_ident(#field_type_range_or_value),

                        });

                        let field_type_fields = Ident::new(&format!("{}Fields", ty_string), variant.ident.span());

                        a.enum_fields.push(quote! {
                            #variant_ident(#field_type_fields)
                        });
                        a.field_names.push(format!("{}", variant_ident));

                        a.range_or_value_add_calls.push(quote! {
                            #range_or_value_struct_ident::#variant_ident(v) => {
                                v.fill_run_configs(|v|append(#fields_enum_ident::#variant_ident(v)), const_params, grid_params, opt_params);
                            }
                        });

                        let field_type_builder = Ident::new(&format!("{}Builder", ty_string), variant_ident.span());
                        let field_ident =  Ident::new(&thoosa_shared::camel_to_snake(&variant_ident.to_token_stream().to_string()), variant.ident.span());
                        a.builder_fields.push(quote! {
                            #field_ident:#field_type_builder,
                        });

                        a.builder_enum_variants.push(quote! {
                            #variant_ident,
                        });

                        a.builder_assignments.push(quote!{
                            // #field_ident: self.#field_ident.build(),
                            #builder_enum_ident::#variant_ident => {
                                #target_struct_ident::#variant_ident(self.#field_ident.build())
                            }
                        });


                        a.builder_add_val_match_cases.push(quote!{
                            #fields_enum_ident::#variant_ident(sub_path) => {
                                let prev = self.#builder_enum_field.replace(#builder_enum_ident::#variant_ident);
                                assert!(prev.as_ref().map(|p|p==&#builder_enum_ident::#variant_ident).unwrap_or(true), "mixed fields from {:?} prev={prev:?}, path={path:?} ", #builder_enum_ident::#variant_ident);

                                self.#field_ident.add_val(sub_path, val);
                            }
                        });
                    }
                }

                a
            });

            token_streams.builder_fields.push(quote! {
                    #builder_enum_field:Option<#builder_enum_ident>,
            });

            let (
                range_or_value_fields,
                enum_fields,
                field_names,
                add_range_or_value_calls,
                builder_fields,
                builder_enum_variants,
                builder_assignments,
                builder_add_val_match_cases
            ) = (token_streams.range_or_value_fields, token_streams.enum_fields, token_streams.field_names, token_streams.range_or_value_add_calls, token_streams.builder_fields, token_streams.builder_enum_variants, token_streams.builder_assignments, token_streams.builder_add_val_match_cases);
            let expect_msg = format!("{} is required", target_struct_ident);
            (
                quote! { #(# range_or_value_fields)* },
                quote! { #(# enum_fields,)* },
                quote! { #(Self::#enum_fields => { write!(f, #field_names)})* },
                quote! {  match self {#(# add_range_or_value_calls)* }},
                quote! {enum},
                quote! { #(# builder_fields)* },
                quote! {
                    match path {
                        #(#builder_add_val_match_cases)*
                    }
                },
                quote! {
                    match self.#builder_enum_field.as_ref().expect(#expect_msg) {
                        #(#builder_assignments)*
                    }
                },
                quote! { #(#builder_enum_variants)*}
            )
        }
        _ => {
            (
                quote! {},
                quote! {},
                quote! {},
                quote! {},
                quote! {},
                quote! {},
                quote! {},
                quote! {},
                quote! {}
            )
        }
    };

    let has_run_config_opt = ast.attrs.iter().any(|attr| {
        attr.path().is_ident("run_config_opt")
    });

    let has_run_config_grid = ast.attrs.iter().any(|attr| {
        attr.path().is_ident("run_config_grid")
    });

    let source_impl = if has_run_config_opt {
        quote! {
            impl thoosa_run_config_args::OptRunConfigSource for #range_or_value_struct_ident {
                type P =  #fields_enum_ident;

                fn opt_run_configs(&self) -> thoosa_shared::run_configs::OptRunConfigs<Vec<thoosa_shared::val::NamedVal<#fields_enum_ident>>, Vec<thoosa_shared::val::NamedRange<#fields_enum_ident>>, Vec<Hyperparameter<#fields_enum_ident>>, #fields_enum_ident> {
                    let mut const_params = Vec::new();
                    let mut grid_params = Vec::new();
                    let mut opt_params = Vec::new();

                    self.fill_run_configs(|v|v, &mut const_params, &mut grid_params, &mut opt_params);

                    thoosa_shared::run_configs::OptRunConfigs::new(
                        thoosa_shared::run_configs::GridRunConfigs::new(
                            thoosa_shared::run_configs::ConstConfigs::new(const_params),
                            grid_params,
                        ),
                        opt_params,
                    )
                }
            }
        }
    } else if has_run_config_grid {
        quote! {
            impl thoosa_run_config_args::GridRunConfigSource for #range_or_value_struct_ident {
                type P =  #fields_enum_ident;

                fn grid_run_configs(&self) -> thoosa_shared::run_configs::GridRunConfigs<Vec<thoosa_shared::val::NamedVal<#fields_enum_ident>>, Vec<thoosa_shared::val::NamedRange<#fields_enum_ident>>> {
                    let mut const_params = Vec::new();
                    let mut grid_params = Vec::new();
                    let mut opt_params = Vec::new();

                    self.fill_run_configs(|v|v, &mut const_params, &mut grid_params, &mut opt_params);
                    if !opt_params.is_empty() {
                        panic!("opt_params in grid");
                    }

                    thoosa_shared::run_configs::GridRunConfigs::new(
                        thoosa_shared::run_configs::ConstConfigs::new(const_params),
                        grid_params,
                    )
                }
            }
        }
    } else {
        quote! {

        }
    };

    let (into_impl, set_id) = if has_run_config_opt || has_run_config_grid {
        (quote! {
            impl <I:IntoIterator<Item=thoosa_shared::val::NamedVal<#fields_enum_ident>>> From<I> for #target_struct_ident {
                fn from(value: I) -> Self {
                    let mut builder = #builder_ident::default();

                    for path_val in value {
                        builder.add_val(path_val.path(), path_val.val());
                    }

                    builder.build()

                }
            }

        }, quote!{
            self.id.replace(id);
        })
    } else {
        (quote! {}, quote! {
            panic!("set id should not be called on run config sub")
        })
    };

    eprintln!("range_or_value_type={}", range_or_value_type.to_string());
    let builder_enum = if "enum" == &range_or_value_type.to_string() {
        eprintln!("BuilderENum");

        quote! {
            #[derive(Debug, PartialEq)]
            enum #builder_enum_ident{
                #builder_enum_variants
            }
        }
    } else {
        quote! {}
    };

    let gen = quote! {

        #[derive(Debug, serde::Deserialize, serde::Serialize)]
        pub #range_or_value_type #range_or_value_struct_ident {
            #range_or_value_fields
        }

        impl thoosa_run_config_args::RunConfigRangeOrValue for #range_or_value_struct_ident {

        }

        impl thoosa_shared::run_configs::RunConfig for #target_struct_ident {

            fn set_id(&mut self, id: thoosa_shared::run_configs::RunId) {
                #set_id
            }

            fn type_name() -> &'static str {
                #target_struct_name
            }
        }

        #[derive(Debug, PartialEq, Clone)]
        pub enum #fields_enum_ident {
            #enum_fields
        }

        //used for hyperparam keys in TPEOptimizer
        impl std::fmt::Display for #fields_enum_ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    #enum_fields_for_display
                }
            }
        }

        impl thoosa_shared::hyperparameter::FieldName for #fields_enum_ident {}

        impl #range_or_value_struct_ident {
            pub fn fill_run_configs<T:thoosa_shared::hyperparameter::FieldName, F>(&self, append: F, const_params: &mut Vec<thoosa_shared::val::NamedVal<T>>, grid_params: &mut Vec<thoosa_shared::val::NamedRange<T>>, opt_params: &mut Vec<thoosa_shared::hyperparameter::Hyperparameter<T>>)
                where F:Fn(#fields_enum_ident)->T{

                #range_or_value_add_calls
            }

        }

        #source_impl

        #[derive(Debug, Default)]
        pub struct #builder_ident {
            #builder_fields
        }

        impl #builder_ident {

            pub fn add_val(&mut self, path: &#fields_enum_ident, val: &thoosa_shared::val::Val) {
                #builder_add_val_match_cases
            }

            pub fn build(&self) -> #target_struct_ident {

                #builder_assignments
            }
        }

        #builder_enum

        #into_impl
    };

    gen.into()
}

