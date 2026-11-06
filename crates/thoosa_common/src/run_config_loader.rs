//! Loading a persisted run configuration and resolving its input configuration.
use log::debug;
use std::{fmt::Debug, sync::Arc};
use thoosa_properties_map::FromHashmap;
use thoosa_shared::run_configs::{RunConfig, RunConfigId};
use thoosa_shared::t;
use thoosa_store::Store;

pub async fn load_run_config<FH, S>(store: &Arc<S>, run_config_id: RunConfigId, run_config_type: &str) -> anyhow::Result<FH>
where
    FH: FromHashmap + RunConfig + Debug,
    S: Store + ?Sized,
{
    let real_run_config_id = if t!(store.run_config_descr(run_config_id)).await?.r#type == run_config_type {
        run_config_id
    } else {
        let run_config_ids = t!(store.get_input_run_config_ids(run_config_id, Some(run_config_type))).await?;
        *run_config_ids.first().ok_or_else(|| anyhow::anyhow!("load_run_config: no {run_config_type} input for find_motifs_run_config_id={run_config_id}"))?
    };

    let map = t!(store.list_run_config_values(real_run_config_id)).await?;
    let mut run_config = FH::from_properties_map(None, &map)?;

    debug!("load_run_config: run_config={run_config:?}");

    run_config.set_id(real_run_config_id);

    Ok(run_config)
}
