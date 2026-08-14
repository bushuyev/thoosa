#![forbid(unsafe_code)]

use std::collections::HashSet;

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use proc_macro_crate::{
    crate_name,
    FoundCrate,
};
use quote::{format_ident, quote};
use syn::{
    parse_macro_input,
    parse_quote,
    spanned::Spanned,
    Attribute,
    Data,
    DataEnum,
    DataStruct,
    DeriveInput,
    Error,
    Field,
    Fields,
    Generics,
    Ident,
    LitStr,
    Member,
    Path,
    Result,
    Type,
    Variant,
};


#[proc_macro_derive(DigestHash, attributes(digest))]
pub fn derive_digest_hash(
    input: TokenStream,
) -> TokenStream {
    let input =
        parse_macro_input!(input as DeriveInput);

    match expand(input) {
        Ok(tokens) => tokens.into(),

        Err(error) => {
            error.to_compile_error().into()
        }
    }
}


fn expand(input: DeriveInput) -> Result<TokenStream2> {
    let runtime = runtime_crate_path()?;

    let DeriveInput {
        ident,
        attrs,
        data,
        mut generics,
        ..
    } = input;

    let type_tag =
        parse_type_tag(&attrs, &ident)?;

    let update = match data {
        Data::Struct(data) => {
            expand_struct(
                data,
                &runtime,
                &type_tag,
                &mut generics,
            )?
        }

        Data::Enum(data) => {
            expand_enum(
                data,
                &runtime,
                &type_tag,
                &mut generics,
            )?
        }

        Data::Union(_) => {
            return Err(Error::new_spanned(
                ident,
                "DigestHash cannot be derived for unions",
            ));
        }
    };

    let (
        impl_generics,
        ty_generics,
        where_clause,
    ) = generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics
            #runtime::DigestHash
            for #ident #ty_generics
            #where_clause
        {
            fn update_digest(
                &self,
                __digest_hasher:
                    &mut #runtime::blake3::Hasher,
            ) {
                #update
            }
        }
    })
}


fn expand_struct(
    data: DataStruct,
    runtime: &TokenStream2,
    type_tag: &LitStr,
    generics: &mut Generics,
) -> Result<TokenStream2> {
    let mut fields = collect_fields(data.fields)?;

    prepare_fields(&mut fields, runtime, generics)?;

    let field_count = fields.fields.len();
    let updates = fields.fields.iter().map(|field| {
        let member = &field.member;
        let value = quote!(&self.#member);

        expand_field_update(field, value, runtime)
    });

    Ok(quote! {
        #runtime::__private::begin_struct(
            __digest_hasher,
            #type_tag,
            #field_count,
        );

        #(#updates)*
    })
}


fn expand_enum(
    data: DataEnum,
    runtime: &TokenStream2,
    type_tag: &LitStr,
    generics: &mut Generics,
) -> Result<TokenStream2> {
    let mut variant_names = HashSet::new();
    let mut arms = Vec::new();

    for variant in data.variants {
        let variant_attrs = parse_variant_attrs(&variant)?;
        let variant_ident = variant.ident;
        let variant_name = variant_attrs.name.unwrap_or_else(|| {
            LitStr::new(
                &variant_ident.to_string(),
                variant_ident.span(),
            )
        });

        if !variant_names.insert(variant_name.value()) {
            return Err(Error::new(
                variant_name.span(),
                format!(
                    "duplicate digest variant name `{}`",
                    variant_name.value(),
                ),
            ));
        }

        let mut fields = collect_fields(variant.fields)?;

        prepare_fields(&mut fields, runtime, generics)?;

        let pattern = enum_variant_pattern(
            &variant_ident,
            &fields,
        );
        let field_count = fields.fields.len();
        let updates = fields.fields.iter().map(|field| {
            let binding = enum_field_binding(&field.member);
            let value = quote!(#binding);

            expand_field_update(field, value, runtime)
        });

        arms.push(quote! {
            #pattern => {
                #runtime::__private::variant(
                    __digest_hasher,
                    #variant_name,
                    #field_count,
                );

                #(#updates)*
            }
        });
    }

    Ok(quote! {
        #runtime::__private::begin_enum(
            __digest_hasher,
            #type_tag,
        );

        match self {
            #(#arms),*
        }
    })
}


fn prepare_fields(
    fields: &mut CollectedFields,
    runtime: &TokenStream2,
    generics: &mut Generics,
) -> Result<()> {
    validate_unique_field_names(fields)?;

    // Named fields are keyed by their stable digest names, so declaration order must not matter.
    // Tuple fields retain their positional order.
    if fields.style == FieldsStyle::Named {
        fields.fields.sort_by(|a, b| {
            a.name
                .value()
                .cmp(&b.name.value())
        });
    }

    // Bound actual field types so wrappers can provide DigestHash without their type parameters doing so.
    for field in &fields.fields {
        if field.with.is_none() {
            let ty = &field.ty;

            generics
                .make_where_clause()
                .predicates
                .push(parse_quote! {
                    #ty: #runtime::DigestHash
                });
        }
    }

    Ok(())
}


fn expand_field_update(
    field: &FieldSpec,
    value: TokenStream2,
    runtime: &TokenStream2,
) -> TokenStream2 {
    let name = &field.name;

    match &field.with {
        Some(with) => {
            quote! {
                #runtime::__private::field(
                    __digest_hasher,
                    #name,
                );

                #with(
                    #value,
                    __digest_hasher,
                );
            }
        }

        None => {
            let ty = &field.ty;

            quote! {
                #runtime::__private::field(
                    __digest_hasher,
                    #name,
                );

                <#ty as #runtime::DigestHash>::update_digest(
                    #value,
                    __digest_hasher,
                );
            }
        }
    }
}


fn enum_variant_pattern(
    variant: &Ident,
    fields: &CollectedFields,
) -> TokenStream2 {
    match fields.style {
        FieldsStyle::Named => {
            let bindings = fields.fields.iter().map(|field| {
                let member = &field.member;
                let binding = enum_field_binding(member);

                quote!(#member: #binding)
            });

            quote!(Self::#variant { #(#bindings,)* .. })
        }

        FieldsStyle::Unnamed => {
            let bindings = (0..fields.total_count).map(|index| {
                let member = Member::Unnamed(syn::Index::from(index));

                if fields.fields.iter().any(|field| {
                    matches!(
                        &field.member,
                        Member::Unnamed(field_index)
                            if field_index.index as usize == index
                    )
                }) {
                    let binding = enum_field_binding(&member);
                    quote!(#binding)
                } else {
                    quote!(_)
                }
            });

            quote!(Self::#variant(#(#bindings),*))
        }

        FieldsStyle::Unit => {
            quote!(Self::#variant)
        }
    }
}


fn enum_field_binding(member: &Member) -> Ident {
    match member {
        Member::Named(ident) => {
            let name = ident.to_string();

            format_ident!(
                "__digest_field_{}",
                name.trim_start_matches("r#"),
                span = ident.span(),
            )
        }

        Member::Unnamed(index) => {
            format_ident!("__digest_field_{}", index.index)
        }
    }
}


// -------------------------------------------------------------------------
// Runtime crate resolution
// -------------------------------------------------------------------------

fn runtime_crate_path() -> Result<TokenStream2> {
    let found =
        crate_name("thoosa_digest_id").map_err(|error| {
            Error::new(
                Span::call_site(),
                format!(
                    "could not find dependency `thoosa_digest_id`: {error}"
                ),
            )
        })?;

    match found {
        FoundCrate::Itself => {
            Ok(quote!(crate))
        }

        FoundCrate::Name(name) => {
            let ident =
                Ident::new(
                    &name,
                    Span::call_site(),
                );

            Ok(quote!(::#ident))
        }
    }
}


// -------------------------------------------------------------------------
// Type and variant attributes
// -------------------------------------------------------------------------

fn parse_type_tag(
    attrs: &[Attribute],
    ident: &Ident,
) -> Result<LitStr> {
    let mut result = None;

    for attr in attrs {
        if !attr.path().is_ident("digest") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("tag") {
                if result.is_some() {
                    return Err(
                        meta.error(
                            "duplicate `tag`",
                        ),
                    );
                }

                let value: LitStr =
                    meta.value()?.parse()?;

                if value.value().is_empty() {
                    return Err(Error::new(
                        value.span(),
                        "digest tag must not be empty",
                    ));
                }

                result = Some(value);

                return Ok(());
            }

            Err(meta.error(
                "supported type attribute: `tag = \"...\"`",
            ))
        })?;
    }

    result.ok_or_else(|| {
        Error::new_spanned(
            ident,
            concat!(
            "DigestHash requires a stable schema tag, for example ",
            "`#[digest(tag = \"my.application.Type:v1\")]`"
            ),
        )
    })
}


#[derive(Default)]
struct VariantAttrs {
    name: Option<LitStr>,
}


fn parse_variant_attrs(
    variant: &Variant,
) -> Result<VariantAttrs> {
    let mut result = VariantAttrs::default();

    for attr in &variant.attrs {
        if !attr.path().is_ident("digest") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                if result.name.is_some() {
                    return Err(
                        meta.error(
                            "duplicate `name`",
                        ),
                    );
                }

                let value: LitStr =
                    meta.value()?.parse()?;

                if value.value().is_empty() {
                    return Err(Error::new(
                        value.span(),
                        "digest variant name must not be empty",
                    ));
                }

                result.name = Some(value);

                return Ok(());
            }

            Err(meta.error(
                "supported variant attribute: `name = \"...\"`",
            ))
        })?;
    }

    Ok(result)
}


// -------------------------------------------------------------------------
// Fields
// -------------------------------------------------------------------------

#[derive(Default)]
struct FieldAttrs {
    skip: bool,
    name: Option<LitStr>,
    with: Option<Path>,
}


fn parse_field_attrs(
    field: &Field,
) -> Result<FieldAttrs> {
    let mut result =
        FieldAttrs::default();

    for attr in &field.attrs {
        if !attr.path().is_ident("digest") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                if result.skip {
                    return Err(
                        meta.error(
                            "duplicate `skip`",
                        ),
                    );
                }

                result.skip = true;

                return Ok(());
            }

            if meta.path.is_ident("name") {
                if result.name.is_some() {
                    return Err(
                        meta.error(
                            "duplicate `name`",
                        ),
                    );
                }

                let value: LitStr =
                    meta.value()?.parse()?;

                if value.value().is_empty() {
                    return Err(Error::new(
                        value.span(),
                        "digest field name must not be empty",
                    ));
                }

                result.name = Some(value);

                return Ok(());
            }

            if meta.path.is_ident("with") {
                if result.with.is_some() {
                    return Err(
                        meta.error(
                            "duplicate `with`",
                        ),
                    );
                }

                let value: LitStr =
                    meta.value()?.parse()?;

                let path: Path =
                    value.parse()?;

                result.with = Some(path);

                return Ok(());
            }

            Err(meta.error(
                concat!(
                "supported field attributes: ",
                "`skip`, `name = \"...\"`, ",
                "`with = \"path::to::function\"`"
                ),
            ))
        })?;
    }

    if result.skip
        && (
        result.name.is_some()
            || result.with.is_some()
    )
    {
        return Err(Error::new(
            field.span(),
            "`skip` cannot be combined with `name` or `with`",
        ));
    }

    Ok(result)
}


struct FieldSpec {
    member: Member,
    ty: Type,
    name: LitStr,
    with: Option<Path>,
}


#[derive(Clone, Copy, PartialEq, Eq)]
enum FieldsStyle {
    Named,
    Unnamed,
    Unit,
}


struct CollectedFields {
    style: FieldsStyle,
    total_count: usize,
    fields: Vec<FieldSpec>,
}


fn collect_fields(
    fields: Fields,
) -> Result<CollectedFields> {
    match fields {
        Fields::Named(fields) => {
            let total_count = fields.named.len();
            let mut result = Vec::new();

            for field in fields.named {
                let attrs =
                    parse_field_attrs(&field)?;

                if attrs.skip {
                    continue;
                }

                let ident =
                    field
                        .ident
                        .clone()
                        .expect(
                            "named field without identifier",
                        );

                let name =
                    attrs.name.unwrap_or_else(|| {
                        LitStr::new(
                            &ident.to_string(),
                            ident.span(),
                        )
                    });

                result.push(FieldSpec {
                    member:
                    Member::Named(ident),

                    ty: field.ty,

                    name,

                    with: attrs.with,
                });
            }

            Ok(CollectedFields {
                style: FieldsStyle::Named,
                total_count,
                fields: result,
            })
        }


        Fields::Unnamed(fields) => {
            let total_count = fields.unnamed.len();
            let mut result = Vec::new();

            for (index, field)
            in fields.unnamed.into_iter().enumerate()
            {
                let attrs =
                    parse_field_attrs(&field)?;

                if attrs.skip {
                    continue;
                }

                let span =
                    field.ty.span();

                let name =
                    attrs.name.unwrap_or_else(|| {
                        LitStr::new(
                            &index.to_string(),
                            span,
                        )
                    });

                result.push(FieldSpec {
                    member:
                    Member::Unnamed(
                        syn::Index::from(index),
                    ),

                    ty: field.ty,

                    name,

                    with: attrs.with,
                });
            }

            Ok(CollectedFields {
                style: FieldsStyle::Unnamed,
                total_count,
                fields: result,
            })
        }


        Fields::Unit => {
            Ok(CollectedFields {
                style: FieldsStyle::Unit,
                total_count: 0,
                fields: Vec::new(),
            })
        }
    }
}


fn validate_unique_field_names(
    fields: &CollectedFields,
) -> Result<()> {
    let mut names =
        HashSet::new();

    for field in &fields.fields {
        let name =
            field.name.value();

        if !names.insert(name.clone()) {
            return Err(Error::new(
                field.name.span(),
                format!(
                    "duplicate digest field name `{name}`",
                ),
            ));
        }
    }

    Ok(())
}
