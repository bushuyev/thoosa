use proc_macro::TokenStream;
use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::{ToTokens, format_ident, quote};
use syn::spanned::Spanned;
use syn::{DeriveInput, parse_macro_input};
use thoosa_macros_utils::*;

const RUN_CONFIG_ATTR: &'static str = "run_config";
// Used on enum variants to define the fallback when no variant selection can be inferred.
const RUN_CONFIG_DEFAULT_ATTR: &'static str = "run_config_default";


/// Derives run-config helper types for Thoosa.
///
/// This macro generates `{TargetStruct}RangeOrValue`, where every field marked
/// with `#[run_config]` is represented as `RangeOrValue`.
///
/// Semantics:
/// - `RangeOrValue::Value` and `RangeOrValue::Range { step: Some(...) }` produce
///   concrete values (constant/grid dimensions).
/// - `RangeOrValue::Range { step: None }` marks optimization fields; their values
///   are selected during optimization runs (`run_opt`, then `AsyncTPEOptimizer`).
///
/// `run_config_opt`/`run_config_grid` sources are generated accordingly and expose
/// `opt_run_configs()` / `grid_run_configs()` helpers.
#[proc_macro_derive(
    ThoosaRunConfig,
    attributes(run_config, run_config_opt, run_config_grid, run_config_default)
)]
pub fn thoosa_args_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    thoosa_args_derive_impl(ast).into()
}

fn thoosa_args_derive_impl(ast: DeriveInput) -> TokenStream2 {

    let target_struct_ident = &ast.ident;
    let target_struct_name = &ast.ident.to_string();

    //eprintln!("thoosa_args_derive: target_struct_ident={target_struct_ident:?}");

    let range_or_value_struct_ident = format_ident!("{}RangeOrValue", target_struct_ident);
    let fields_enum_ident = format_ident!("{}Fields", target_struct_ident);
    let builder_ident = format_ident!("{}Builder", target_struct_ident);
    let builder_enum_ident = format_ident!("{}BuilderEnum", target_struct_ident);
    let vec_index_ty_ident = format_ident!("{}VecIndex", fields_enum_ident);

    struct TokenStreams {
        range_or_value_fields: Vec<proc_macro2::TokenStream>,
        range_or_value_add_calls: Vec<proc_macro2::TokenStream>,
        enum_fields: Vec<proc_macro2::TokenStream>,
        field_names: Vec<String>,

        builder_fields: Vec<proc_macro2::TokenStream>,
        builder_add_val_match_cases: Vec<proc_macro2::TokenStream>,
        builder_assignments: Vec<proc_macro2::TokenStream>,
        builder_enum_variants: Vec<proc_macro2::TokenStream>,
        has_vec_indexes: bool,
    }

    impl TokenStreams {
        pub fn new() -> Self {
            Self {
                range_or_value_fields: vec![],
                range_or_value_add_calls: vec![],
                enum_fields: vec![],
                field_names: vec![],

                builder_fields: vec![],
                builder_add_val_match_cases: vec![],
                builder_assignments: vec![],
                builder_enum_variants: vec![],
                has_vec_indexes: false,
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
        vec_index_alias,
    ) = match &ast.data {
        syn::Data::Struct(data_struct) => {
            match &data_struct.fields {
                syn::Fields::Named(named_fields) => {
                    let  token_streams = named_fields.named.iter().fold(TokenStreams::new(), |mut a, field| {
                        let field_ident = field.ident.as_ref().unwrap();
                        let camel_field_ident = Ident::new(&thoosa_shared::snake_to_camel(&field_ident.to_string()), fields_enum_ident.span());
                        let field_type = field.ty.clone();

                        //eprintln!("field_type={}", field_type.to_token_stream());

                        if has_attr(field, RUN_CONFIG_ATTR) {
                            if let Some(inner_ty) = get_option_inner_ty(&field_type) {//is Option
                                if let Some(vec_inner_ty) = get_vec_inner_ty(&inner_ty) {
                                    a.has_vec_indexes = true;
                                    let vec_inner_ty_str = vec_inner_ty.to_token_stream().to_string();

                                    if is_primitive(&vec_inner_ty) {
                                        a.range_or_value_fields.push(quote! {
                                            pub #field_ident: Option<Vec<thoosa_shared::RangeOrValue<#vec_inner_ty>>>,
                                        });

                                        a.enum_fields.push(quote! {
                                            #camel_field_ident(#vec_index_ty_ident)
                                        });
                                        a.field_names.push(field_ident.to_string());

                                        a.range_or_value_add_calls.push(quote! {
                                            if let Some(vs) = self.#field_ident.as_ref() {
                                                for (i, v) in vs.iter().enumerate() {
                                                    thoosa_shared::add_range_or_value::<#vec_inner_ty, T>(append(#fields_enum_ident::#camel_field_ident(i)), v, const_params, grid_params, opt_params);
                                                }
                                            }
                                        });

                                        a.builder_fields.push(quote! {
                                            #field_ident:std::collections::BTreeMap<usize, #vec_inner_ty>,
                                        });

                                        let field_ident_str = field_ident.to_string();
                                        a.builder_add_val_match_cases.push(quote! {
                                            #fields_enum_ident::#camel_field_ident(i) => {
                                                let v:#vec_inner_ty = val.try_into().expect(&format!("can't convert val {:?}:{}[{}] to {}", val, #field_ident_str, i, #vec_inner_ty_str));
                                                self.#field_ident.insert(*i, v);
                                            }
                                        });

                                        a.builder_assignments.push(quote!{
                                            #field_ident: {
                                                if self.#field_ident.is_empty() {
                                                    None
                                                } else {
                                                    let mut v = Vec::with_capacity(self.#field_ident.len());
                                                    for (expected_idx, (idx, item)) in self.#field_ident.iter().enumerate() {
                                                        assert_eq!(*idx, expected_idx, "sparse vec for {}: expected index {} got {}", #field_ident_str, expected_idx, idx);
                                                        v.push(item.clone());
                                                    }
                                                    Some(v)
                                                }
                                            },
                                        });
                                    } else {
                                        let field_type_range_or_value = Ident::new(&format!("{}RangeOrValue", vec_inner_ty.to_token_stream().to_string()), camel_field_ident.span());
                                        let field_type_fields = Ident::new(&format!("{}Fields", vec_inner_ty.to_token_stream().to_string()), camel_field_ident.span());
                                        let field_type_builder = Ident::new(&format!("{}Builder", vec_inner_ty.to_token_stream().to_string()), camel_field_ident.span());

                                        a.range_or_value_fields.push(quote! {
                                            pub #field_ident: Option<Vec<#field_type_range_or_value>>,
                                        });

                                        a.enum_fields.push(quote! {
                                            #camel_field_ident(#vec_index_ty_ident, #field_type_fields)
                                        });
                                        a.field_names.push(field_ident.to_string());

                                        a.range_or_value_add_calls.push(quote! {
                                            if let Some(vs) = self.#field_ident.as_ref() {
                                                for (i, v) in vs.iter().enumerate() {
                                                    v.fill_run_configs(|sub| append(#fields_enum_ident::#camel_field_ident(i, sub)), const_params, grid_params, opt_params);
                                                }
                                            }
                                        });

                                        a.builder_fields.push(quote! {
                                            #field_ident:std::collections::BTreeMap<usize, #field_type_builder>,
                                        });

                                        a.builder_add_val_match_cases.push(quote! {
                                            #fields_enum_ident::#camel_field_ident(i, sub_path) => {
                                                self.#field_ident.entry(*i).or_default().add_val(sub_path, val);
                                            }
                                        });

                                        let field_ident_str = field_ident.to_string();
                                        a.builder_assignments.push(quote!{
                                            #field_ident: {
                                                if self.#field_ident.is_empty() {
                                                    None
                                                } else {
                                                    let mut v = Vec::with_capacity(self.#field_ident.len());
                                                    for (expected_idx, (idx, item)) in self.#field_ident.iter().enumerate() {
                                                        assert_eq!(*idx, expected_idx, "sparse vec for {}: expected index {} got {}", #field_ident_str, expected_idx, idx);
                                                        v.push(item.build());
                                                    }
                                                    Some(v)
                                                }
                                            },
                                        });
                                    }
                                } else if is_primitive(&inner_ty) {
                                    a.range_or_value_fields.push(quote! {
                                        pub #field_ident: Option<thoosa_shared::RangeOrValue<#inner_ty>>,
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
                                        pub #field_ident: Option<#field_type_range_or_value>,
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
                                if let Some(vec_inner_ty) = get_vec_inner_ty(&field_type) {
                                    a.has_vec_indexes = true;
                                    let vec_inner_ty_str = vec_inner_ty.to_token_stream().to_string();

                                    if is_primitive(&vec_inner_ty) {
                                        a.range_or_value_fields.push(quote! {
                                            pub #field_ident: Vec<thoosa_shared::RangeOrValue<#vec_inner_ty>>,
                                        });
                                        a.enum_fields.push(quote! {
                                            #camel_field_ident(#vec_index_ty_ident)
                                        });
                                        a.field_names.push(field_ident.to_string());

                                        a.range_or_value_add_calls.push(quote! {
                                            for (i, v) in self.#field_ident.iter().enumerate() {
                                                thoosa_shared::add_range_or_value::<#vec_inner_ty, T>(append(#fields_enum_ident::#camel_field_ident(i)), v, const_params, grid_params, opt_params);
                                            }
                                        });

                                        a.builder_fields.push(quote! {
                                            #field_ident:std::collections::BTreeMap<usize, #vec_inner_ty>,
                                        });

                                        let field_ident_str = field_ident.to_string();
                                        a.builder_add_val_match_cases.push(quote! {
                                            #fields_enum_ident::#camel_field_ident(i) => {
                                                let v:#vec_inner_ty = val.try_into().expect(&format!("can't convert val {:?}:{}[{}] to {}", val, #field_ident_str, i, #vec_inner_ty_str));
                                                self.#field_ident.insert(*i, v);
                                            }
                                        });

                                        let expect_msg = format!("no {}", field_ident);
                                        a.builder_assignments.push(quote!{
                                            #field_ident: {
                                                assert!(!self.#field_ident.is_empty(), #expect_msg);
                                                let mut v = Vec::with_capacity(self.#field_ident.len());
                                                for (expected_idx, (idx, item)) in self.#field_ident.iter().enumerate() {
                                                    assert_eq!(*idx, expected_idx, "sparse vec for {}: expected index {} got {}", #field_ident_str, expected_idx, idx);
                                                    v.push(item.clone());
                                                }
                                                v
                                            },
                                        });
                                    } else {
                                        let field_type_range_or_value = Ident::new(&format!("{}RangeOrValue", vec_inner_ty.to_token_stream().to_string()), camel_field_ident.span());
                                        let field_type_fields = Ident::new(&format!("{}Fields", vec_inner_ty.to_token_stream().to_string()), camel_field_ident.span());
                                        let field_type_builder = Ident::new(&format!("{}Builder", vec_inner_ty.to_token_stream().to_string()), camel_field_ident.span());

                                        a.range_or_value_fields.push(quote! {
                                            pub #field_ident: Vec<#field_type_range_or_value>,
                                        });

                                        a.enum_fields.push(quote! {
                                            #camel_field_ident(#vec_index_ty_ident, #field_type_fields)
                                        });
                                        a.field_names.push(field_ident.to_string());

                                        a.range_or_value_add_calls.push(quote! {
                                            for (i, v) in self.#field_ident.iter().enumerate() {
                                                v.fill_run_configs(|sub| append(#fields_enum_ident::#camel_field_ident(i, sub)), const_params, grid_params, opt_params);
                                            }
                                        });

                                        a.builder_fields.push(quote! {
                                            #field_ident:std::collections::BTreeMap<usize, #field_type_builder>,
                                        });

                                        a.builder_add_val_match_cases.push(quote! {
                                            #fields_enum_ident::#camel_field_ident(i, sub_path) => {
                                                self.#field_ident.entry(*i).or_default().add_val(sub_path, val);
                                            }
                                        });

                                        let expect_msg = format!("no {}", field_ident);
                                        let field_ident_str = field_ident.to_string();
                                        a.builder_assignments.push(quote!{
                                            #field_ident: {
                                                assert!(!self.#field_ident.is_empty(), #expect_msg);
                                                let mut v = Vec::with_capacity(self.#field_ident.len());
                                                for (expected_idx, (idx, item)) in self.#field_ident.iter().enumerate() {
                                                    assert_eq!(*idx, expected_idx, "sparse vec for {}: expected index {} got {}", #field_ident_str, expected_idx, idx);
                                                    v.push(item.build());
                                                }
                                                v
                                            },
                                        });
                                    }
                                } else if is_primitive(&field_type) {
                                    a.range_or_value_fields.push(quote! {
                                        pub #field_ident: thoosa_shared::RangeOrValue<#field_type>,
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
                                        pub #field_ident: #field_type_range_or_value,
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
                                pub #field_ident: #field_type,
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

                                // keep for clearer error in expect below
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
                    let is_empty_struct = field_names.is_empty();
                    let has_vec_indexes = token_streams.has_vec_indexes;
                    let empty_marker_ident = format_ident!("__Empty");
                    let enum_fields_tokens = if is_empty_struct {
                        quote! { #empty_marker_ident, }
                    } else {
                        quote! { #(# enum_fields,)* }
                    };
                    let enum_fields_for_display = if is_empty_struct {
                        quote! { Self::#empty_marker_ident => f.write_str("") }
                    } else {
                        quote! { #( Self::#enum_fields => { write!(f, #field_names)})* }
                    };
                    let builder_add_val_match_cases = if is_empty_struct {
                        quote! { let _ = (path, val); }
                    } else {
                        quote! { match path { #(# add_val_match_cases)*} }
                    };
                    let range_or_value_add_calls = if is_empty_struct {
                        quote! {
                            const_params.push(thoosa_shared::val::NamedVal::from((
                                append(#fields_enum_ident::#empty_marker_ident),
                                thoosa_shared::val::Val::Bool(true),
                            )));
                        }
                    } else {
                        quote! { #(# add_range_or_value_calls)* }
                    };

                    (
                        quote! { #(# range_or_value_fields)* },
                        enum_fields_tokens,
                        enum_fields_for_display,
                        range_or_value_add_calls,
                        quote! {struct},
                        quote! { #(# builder_fields)* },
                        builder_add_val_match_cases,
                        quote! {

                            #target_struct_ident {
                                #(# builder_assignments)*
                            }
                        },
                        quote! {},
                        if has_vec_indexes {
                            quote! { type #vec_index_ty_ident = usize; }
                        } else {
                            quote! {}
                        },
                    )
                }
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
                        quote! {},
                        quote! {},
                    )
                }
            }
        }
        syn::Data::Enum(data_enum) => {
            // NOTE: Enum builder selection is driven by "paths" seen in add_val calls.
            // When a variant's inner config emits no params (e.g., empty struct or all Option=None),
            // we emit a synthetic marker field (<Variant>__Selected) to force a path and lock the variant.
            // This prevents falling back to the default variant when a specific enum variant was chosen.
            let builder_enum_field = Ident::new(
                &thoosa_shared::camel_to_snake(&builder_enum_ident.to_string()),
                data_enum.variants.span(),
            );
            let mut default_variant: Option<(Ident, syn::Type)> = None;
            let mut default_variant_error: Option<syn::Error> = None;
            let mut token_streams = TokenStreams::new();
            for variant in data_enum.variants.iter() {
                let variant_ident = &variant.ident;

                if let syn::Fields::Unnamed(fields) = &variant.fields {
                    if fields.unnamed.len() == 1 {
                        let ty = &fields.unnamed.first().expect("no config type").ty;
                        if variant
                            .attrs
                            .iter()
                            .any(|attr| attr.path().is_ident(RUN_CONFIG_DEFAULT_ATTR))
                        {
                            // Only one default is allowed; it is used when *no* variant was inferred.
                            if default_variant.is_some() {
                                default_variant_error = Some(syn::Error::new_spanned(
                                    variant_ident,
                                    "run_config_default can only be set on one variant",
                                ));
                            } else {
                                default_variant = Some((variant_ident.clone(), ty.clone()));
                            }
                        }

                        let ty_string = ty.to_token_stream().to_string();
                        let field_type_range_or_value =
                            Ident::new(&format!("{}RangeOrValue", ty_string), variant.ident.span());

                        token_streams.range_or_value_fields.push(quote! {
                            #variant_ident(#field_type_range_or_value),

                        });

                        let field_type_fields =
                            Ident::new(&format!("{}Fields", ty_string), variant.ident.span());

                        // Synthetic marker for "this enum variant was chosen" when no fields are emitted.
                        let marker_ident = Ident::new(&format!("{}__Selected", variant_ident), variant.ident.span());

                        token_streams.enum_fields.push(quote! {
                            #variant_ident(#field_type_fields)
                        });
                        token_streams.field_names.push(format!("{}", variant_ident));
                        token_streams.enum_fields.push(quote! {
                            #marker_ident
                        });
                        token_streams
                            .field_names
                            .push(format!("{}.__selected", variant_ident));

                        token_streams.range_or_value_add_calls.push(quote! {
                            #range_or_value_struct_ident::#variant_ident(v) => {
                                let added_any = std::cell::Cell::new(false);
                                v.fill_run_configs(
                                    |v| {
                                        added_any.set(true);
                                        append(#fields_enum_ident::#variant_ident(v))
                                    },
                                    const_params,
                                    grid_params,
                                    opt_params,
                                );
                                // If the inner config produced no params, emit the marker so the builder
                                // can still infer the selected enum variant.
                                if !added_any.get() {
                                    const_params.push(thoosa_shared::val::NamedVal::from((
                                        append(#fields_enum_ident::#marker_ident),
                                        thoosa_shared::val::Val::Bool(true),
                                    )));
                                }
                            }
                        });

                        let field_type_builder =
                            Ident::new(&format!("{}Builder", ty_string), variant_ident.span());
                        let field_ident = Ident::new(
                            &thoosa_shared::camel_to_snake(&variant_ident.to_token_stream().to_string()),
                            variant.ident.span(),
                        );
                        token_streams.builder_fields.push(quote! {
                            #field_ident:#field_type_builder,
                        });

                        token_streams.builder_enum_variants.push(quote! {
                            #variant_ident,
                        });

                        token_streams.builder_assignments.push(quote!{
                            Some(#builder_enum_ident::#variant_ident) => {
                                #target_struct_ident::#variant_ident(self.#field_ident.build())
                            }
                        });

                        token_streams.builder_add_val_match_cases.push(quote!{
                            #fields_enum_ident::#variant_ident(sub_path) => {
                                let prev = self.#builder_enum_field.replace(#builder_enum_ident::#variant_ident);
                                assert!(prev.as_ref().map(|p|p==&#builder_enum_ident::#variant_ident).unwrap_or(true), "mixed fields from {:?} prev={prev:?}, path={path:?} ", #builder_enum_ident::#variant_ident);

                                self.#field_ident.add_val(sub_path, val);
                            }
                        });
                        token_streams.builder_add_val_match_cases.push(quote!{
                            #fields_enum_ident::#marker_ident => {
                                let prev = self.#builder_enum_field.replace(#builder_enum_ident::#variant_ident);
                                assert!(prev.as_ref().map(|p|p==&#builder_enum_ident::#variant_ident).unwrap_or(true), "mixed fields from {:?} prev={prev:?}, path={path:?} ", #builder_enum_ident::#variant_ident);
                                let _ = val;
                            }
                        });
                    }
                }
            }

            if let Some(err) = default_variant_error {
                return err.to_compile_error().into();
            }

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
                builder_add_val_match_cases,
            ) = (
                token_streams.range_or_value_fields,
                token_streams.enum_fields,
                token_streams.field_names,
                token_streams.range_or_value_add_calls,
                token_streams.builder_fields,
                token_streams.builder_enum_variants,
                token_streams.builder_assignments,
                token_streams.builder_add_val_match_cases,
            );
            let expect_msg = format!("{} is required", target_struct_ident);
            // Fallback is only used when no variant path was observed at all.
            // This keeps explicit variant selection (including marker paths) authoritative.
            let default_variant_fallback = default_variant.map(|(variant_ident, variant_ty)| {
                quote! {
                    #target_struct_ident::#variant_ident(<#variant_ty as std::default::Default>::default())
                }
            });
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
                if let Some(default_fallback) = default_variant_fallback {
                    quote! {
                        match self.#builder_enum_field.as_ref() {
                            #(#builder_assignments)*
                            None => { #default_fallback }
                        }
                    }
                } else {
                    quote! {
                        match self.#builder_enum_field.as_ref() {
                            #(#builder_assignments)*
                            None => { panic!(#expect_msg) }
                        }
                    }
                },
                quote! { #(#builder_enum_variants)*},
                quote! {},
            )
        }
        _ => (
            quote! {},
            quote! {},
            quote! {},
            quote! {},
            quote! {},
            quote! {},
            quote! {},
            quote! {},
            quote! {},
            quote! {},
        ),
    };

    let has_run_config_opt = ast
        .attrs
        .iter()
        .any(|attr| attr.path().is_ident("run_config_opt"));

    let has_run_config_grid = ast
        .attrs
        .iter()
        .any(|attr| attr.path().is_ident("run_config_grid"));

    let source_impl = if has_run_config_opt {
        quote! {
            impl thoosa_run_config_args::OptRunConfigSource for #range_or_value_struct_ident {
                type P =  #fields_enum_ident;

                fn opt_run_configs(&self) -> Result<thoosa_shared::run_configs::OptRunConfigs<Vec<thoosa_shared::val::NamedVal<#fields_enum_ident>>, Vec<thoosa_shared::val::NamedRange<#fields_enum_ident>>, Vec<thoosa_shared::hyperparameter::Hyperparameter<#fields_enum_ident>>, #fields_enum_ident>, thoosa_shared::ThoError> {
                    let mut const_params = Vec::new();
                    let mut grid_params = Vec::new();
                    let mut opt_params = Vec::new();

                    self.fill_run_configs(|v|v, &mut const_params, &mut grid_params, &mut opt_params);
                    if opt_params.is_empty() {
                        return Err(thoosa_shared::ThoError::new("no opt params in run config"));
                    }

                    Ok(thoosa_shared::run_configs::OptRunConfigs::new(
                        thoosa_shared::run_configs::GridRunConfigs::new(
                            thoosa_shared::run_configs::ConstConfigs::new(const_params),
                            grid_params,
                        ),
                        opt_params,
                    ))
                }
            }
        }
    } else if has_run_config_grid {
        quote! {
            impl thoosa_run_config_args::GridRunConfigSource for #range_or_value_struct_ident {
                type P =  #fields_enum_ident;

                fn grid_run_configs(&self) -> Result<thoosa_shared::run_configs::GridRunConfigs<Vec<thoosa_shared::val::NamedVal<#fields_enum_ident>>, Vec<thoosa_shared::val::NamedRange<#fields_enum_ident>>>, thoosa_shared::ThoError> {
                    let mut const_params = Vec::new();
                    let mut grid_params = Vec::new();
                    let mut opt_params = Vec::new();

                    self.fill_run_configs(|v|v, &mut const_params, &mut grid_params, &mut opt_params);
                    if !opt_params.is_empty() {
                        return Err(thoosa_shared::ThoError::new("opt params in grid run config"));
                    }

                    Ok(thoosa_shared::run_configs::GridRunConfigs::new(
                        thoosa_shared::run_configs::ConstConfigs::new(const_params),
                        grid_params,
                    ))
                }
            }
        }
    } else {
        quote! {}
    };

    let (into_impl, set_id) = if has_run_config_opt || has_run_config_grid {
        (
            quote! {
                impl <I:IntoIterator<Item=thoosa_shared::val::NamedVal<#fields_enum_ident>>> From<I> for #target_struct_ident {
                    fn from(value: I) -> Self {
                        let mut builder = #builder_ident::default();

                        for path_val in value {
                            builder.add_val(path_val.path(), path_val.val());
                        }

                        builder.build()

                    }
                }

            },
            quote! {
                self.id.replace(id);
            },
        )
    } else {
        (
            quote! {},
            quote! {
                panic!("set id should not be called on run config sub")
            },
        )
    };

    //eprintln!("range_or_value_type={}", range_or_value_type.to_string());
    let builder_enum = if "enum" == &range_or_value_type.to_string() {
        //eprintln!("BuilderENum");

        quote! {
            #[derive(Debug, PartialEq)]
            enum #builder_enum_ident{
                #builder_enum_variants
            }
        }
    } else {
        quote! {}
    };

    let token_stream = quote! {

        #[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
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

        #vec_index_alias

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

    token_stream
}

#[cfg(test)]
mod tests {
    use super::{thoosa_args_derive_impl};
    use quote::quote;

    #[test]
    fn emits_enum_variant_marker_when_inner_has_no_params() {
        let input = quote! {
            pub enum Choice {
                #[run_config_default]
                A(EmptyConfig),
                B(EmptyConfig),
            }
        };

        let ast = syn::parse2::<syn::DeriveInput>(input).expect("derive input");
        let output = thoosa_args_derive_impl(ast).to_string();

        assert!(output.contains("A__Selected"));
        assert!(output.contains("B__Selected"));
        assert!(output.contains("added_any"));
        assert!(output.contains("Val :: Bool"));
    }
}
