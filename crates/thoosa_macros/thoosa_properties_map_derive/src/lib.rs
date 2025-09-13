use proc_macro::TokenStream;
use log::warn;
use quote::{format_ident, quote, ToTokens};
use syn::{parse_macro_input, Data, DeriveInput};
use thoosa_macros_utils::{get_option_inner_ty, has_attr, is_primitive};

const IN_MAP_ATTR: &'static str = "in_map";

#[proc_macro_derive(ThoosaPropertiesMap, attributes(in_map))]
pub fn thoosa_args_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let target_ident = &ast.ident;
    let target_ident_prefix = format!("{}::", ast.ident);

    //eprintln!("thoosa_args_derive: target_struct_ident={target_ident:?}");

    match &ast.data {
        syn::Data::Struct(data_struct) => {

            match &data_struct.fields {
                syn::Fields::Named(named_fields) => {

                    let mut fields_in_to_properties = vec![];
                    let mut chains_in_to_properties = vec![];
                    let mut fields_in_from_properties = vec![];

                    for field in named_fields.named.iter().filter(|field| has_attr(field, IN_MAP_ATTR)) {
                        let field_ident = field.ident.as_ref().expect("Should be an identifier");
                        let field_name = format!("{}", field_ident);
                        let field_name_path = format!("{}.", field_ident);

                        if let Some(inner_ty) = get_option_inner_ty(&field.ty) { //is Option

                            if is_primitive(&inner_ty) {
                                fields_in_to_properties.push(quote! {
                                    (thoosa_shared::conc_str(&prefix, #field_name), self.#field_ident.map(|v| v.into())),

                                });

                                fields_in_from_properties.push(quote! {
                                      #field_ident: properties.get(&thoosa_shared::conc_str(&prefix, #field_name))
                                                .map(|v|v.try_into().expect(&format!("can't convert {}: value={v:?}", #field_name))),


                                });
                            } else {
                                let field_type = inner_ty.clone();
                                let _field_path = format!("{}::{}", field_type.to_token_stream(), field_name_path);

                                chains_in_to_properties.push(quote! {
                                    .chain(self.#field_ident.as_ref().map(|s|s.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, #field_name_path)))).unwrap_or(Box::new(std::iter::empty::<(String, thoosa_shared::val::Val)>())))

                                });

                                fields_in_from_properties.push(quote! {
                                    #field_ident: #field_type::from_properties_map(Some(thoosa_shared::conc_str(&prefix, #field_name_path)), &properties).ok(),

                                });
                            }

                        } else {
                            if is_primitive(&field.ty) {
                                fields_in_to_properties.push(quote! {
                                    (thoosa_shared::conc_str(&prefix, #field_name), Some(self.#field_ident.into())),

                                });

                                fields_in_from_properties.push(quote! {
                                    #field_ident: properties.get(&thoosa_shared::conc_str(&prefix, #field_name))
                                        .map(|v|v.try_into().expect(&format!("can't convert {}: value={v:?}", #field_name)))
                                        .ok_or_else(||thoosa_shared::ThoosaError::new(format!("no {} in properties", thoosa_shared::conc_str(&prefix, #field_name))))?,

                                });

                            } else {
                                let field_type = field.ty.clone();
                                let _field_path = format!("{}::{}", field_type.to_token_stream(), field_name_path);
                                chains_in_to_properties.push(quote! {
                                    .chain(self.#field_ident.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, #field_name_path))))
                                });

                                fields_in_from_properties.push(quote! {
                                    #field_ident: #field_type::from_properties_map(Some(thoosa_shared::conc_str(&prefix, #field_name_path)), &properties)?,
                                });
                            }

                        }
                    }


                    let token_stream = quote! {
                        impl thoosa_properties_map::ToHashmap for #target_ident {
                            fn to_properties_iter(&self, parent: Option<String>,) -> Box<dyn Iterator<Item=(String, thoosa_shared::val::Val)>> {
                                let prefix = parent.unwrap_or(#target_ident_prefix.to_string());
                                Box::new(
                                    [
                                        #(#fields_in_to_properties)*
                                    ]
                                    .into_iter()
                                    .filter_map(|v|v.1.map(|val|(v.0, val)))
                                    #(#chains_in_to_properties)*
                                )
                            }
                        }

                        impl thoosa_properties_map::FromHashmap for #target_ident {
                            fn from_properties_map(parent:Option<String>, properties: &std::collections::HashMap<String, thoosa_shared::val::Val>) -> Result<Self, thoosa_shared::ThoosaError> {
                                let prefix = parent.unwrap_or(#target_ident_prefix.to_string());

                                Ok(#target_ident {
                                    #(#fields_in_from_properties)*
                                })
                            }

                        }
                    };

                    token_stream.into()
                }
                _ => {
                    quote!{}.into()
                }
            }
        }
        Data::Enum(data_enum) => {
            let mut variants_in_to_properties = vec![];
            let mut variants_in_from_properties = vec![];

            for variant in &data_enum.variants {
                let variant_ident = &variant.ident;
                let variant_name = thoosa_shared::camel_to_snake(&variant_ident.to_token_stream().to_string());
                let variant_path = thoosa_shared::conc_str(&variant_name, ".");
                if let syn::Fields::Unnamed(fields) = &variant.fields {
                    if fields.unnamed.len() == 1 {
                        //TODO proper Type to Ident
                        let ty = format_ident!("{}", &fields.unnamed.first().expect("no config type").ty.to_token_stream().to_string());
                        variants_in_to_properties.push(quote! {
                            #target_ident::#variant_ident(v) => {
                                Box::new([
                                    (thoosa_shared::conc_str(&prefix, #variant_name), thoosa_shared::val::Val::Bool(true)),
                                ]
                                .into_iter()
                                .chain(
                                    v.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, #variant_path)))))
                            }
                        });

                        if !variants_in_from_properties.is_empty() {
                            variants_in_from_properties.push(quote! {
                                else
                            });
                        }

                        variants_in_from_properties.push(quote! {
                            if properties.get(&thoosa_shared::conc_str(&prefix, #variant_name)).is_some() {
                                Ok(#target_ident::#variant_ident(#ty::from_properties_map(Some(thoosa_shared::conc_str(&prefix, #variant_path)), properties)?))

                            }
                        });


                    } else {
                        warn!("Can't handle more than one field")
                    }
                }

            }

            variants_in_from_properties.push(quote! {
                else {
                    Err(thoosa_shared::ThoosaError::new(format!("Unexpected variant prefix={prefix} in {properties:?} ")))
                }
            });

            quote!{
                 impl thoosa_properties_map::ToHashmap for #target_ident {
                    fn to_properties_iter(&self, parent: Option<String>) -> Box<dyn Iterator<Item=(String, thoosa_shared::val::Val)>> {
                        let prefix = parent.unwrap_or(#target_ident_prefix.to_string());
                        match &self {
                            #(#variants_in_to_properties)*
                        }
                    }
                }

                impl thoosa_properties_map::FromHashmap for #target_ident {

                    fn from_properties_map(parent:Option<String>, properties: &std::collections::HashMap<String, thoosa_shared::val::Val>) -> Result<Self, thoosa_shared::ThoosaError>{
                        log::debug!("{}from_properties_map: parent={parent:?} properties={properties:?}", #target_ident_prefix);

                        let prefix = parent.unwrap_or(#target_ident_prefix.to_string());
                        #(#variants_in_from_properties)*
                    }
                }
            }.into()
        }
        _ => {
            quote!{}.into()
        }
    }


}
