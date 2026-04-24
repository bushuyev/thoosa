use crate::ThoError;
use crate::applicables_runner::Input;
use crate::traits::WithId;
use async_trait::async_trait;
use log::trace;
use mockall::automock;
use rand::rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;

#[automock]
#[async_trait]
pub trait ResourceStore {
    async fn list_resources_by_file_stems<'a>(
        &self,
        file_stems: &'a [&'a str],
    ) -> Result<Vec<Resource>, ThoError>;

    async fn list_resource_set_resources_by_descr(
        &self,
        descr: &str,
    ) -> Result<Vec<Resource>, ThoError>;

    async fn list_resource_set_group_sets(
        &self,
        group_descr: &str,
    ) -> Result<Vec<(usize, String)>, ThoError>;

    async fn list_group_labeled_resources_sets(
        &self,
        group_decr: &str,
    ) -> Result<Vec<(String, String)>, ThoError>;

    async fn list_resource_set_resources(&self, set_id: usize) -> Result<Vec<Resource>, ThoError>;

    async fn list_resource_random_in_group(
        &self,
        group_descr: &str,
    ) -> Result<Vec<(Resource, Vec<f32>)>, ThoError>;

    async fn list_resources_not_in_group(
        &self,
        group_descr: &str,
        limit: u32,
    ) -> Result<Vec<Resource>, ThoError>;

    async fn list_random_resources(
        &self,
        limit: Option<u32>,
    ) -> Result<Vec<Resource>, ThoError>;

}

#[derive(Debug, Clone, DummyData, Named)]
#[name = "n/a"]
#[dummy_data(clone)]
pub struct Resource {
    id: ResourceId,
    import_id: u32,
    file_stem: String,
    length: u32,
}

impl Resource {
    pub fn new<I:Into<String>+Display>(id: u32, import_id: u32, file_stem: I, length: u32) -> Self {
        trace!("Resource::new: id={id} import_id={import_id} file_stem={file_stem}");

        Self {
            id,
            import_id,
            file_stem:file_stem.into(),
            length,

        }
    }

    pub fn id(&self) -> ResourceId {
        self.id
    }

    pub fn import_id(&self) -> u32 {
        self.import_id
    }

    pub fn file_stem(&self) -> &str {
        self.file_stem.as_str()
    }

    pub fn length(&self) -> u32 {
        self.length
    }




}


impl WithId for Resource {
    type ID = Resource;

    fn id(&self) -> &Self::ID {
        &self
    }//return composite id+import_id?
}

impl Input for Resource {
    
}

impl From<(u32, u32, &str, u32)> for Resource {
    fn from(value: (u32, u32, &str, u32)) -> Self {
        Resource::new(value.0, value.1, value.2.to_string(), value.3)
    }
}

impl Hash for Resource {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl PartialEq<Self> for Resource {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}

impl Eq for Resource {}

impl PartialOrd<Self> for Resource {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id().partial_cmp(&other.id)
    }
}

impl Ord for Resource {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

pub type ResourceId = u32;

impl AsRef<ResourceId> for Resource {
    fn as_ref(&self) -> &ResourceId {
        &self.id
    }
}

/// Builds a per-label train/valid map and, for each label, a same-sized shuffled sample of
/// train/valid resources belonging to all other labels.
pub fn build_resource_maps(
    train_labels_resource_keys: Vec<(String, Vec<Resource>)>,
    valid_label_resource_keys: Vec<(String, Vec<Resource>)>,
) -> (HashMap<String, (Vec<Resource>, Vec<Resource>)>, Arc<HashMap<String, (Vec<Resource>, Vec<Resource>)>>) {
    let mut rand = rng();
    build_resource_maps_with_rng(train_labels_resource_keys, valid_label_resource_keys, &mut rand)
}

fn build_resource_maps_with_rng<R: rand::Rng + ?Sized>(
    train_labels_resource_keys: Vec<(String, Vec<Resource>)>,
    valid_label_resource_keys: Vec<(String, Vec<Resource>)>,
    rng: &mut R,
) -> (HashMap<String, (Vec<Resource>, Vec<Resource>)>, Arc<HashMap<String, (Vec<Resource>, Vec<Resource>)>>) {
    assert_eq!(train_labels_resource_keys.len(), valid_label_resource_keys.len(), "train_labels_resource_keys={train_labels_resource_keys:?} valid_label_resource_keys={valid_label_resource_keys:?}");

    let mut labels_others_train_valid_resources: HashMap<String, (Vec<&Resource>, Vec<&Resource>)> =
        train_labels_resource_keys.iter().fold(HashMap::new(), |mut acc, entry| {
            assert!(acc.insert(entry.0.clone(), (vec![], vec![])).is_none());
            acc
        });

    let labels_train_and_valid_sets = train_labels_resource_keys
        .into_iter()
        .zip(valid_label_resource_keys)
        .fold(HashMap::new(), |mut acc, (train, valid)| {
            assert_eq!(train.0, valid.0);
            assert!(acc.insert(train.0, (train.1, valid.1)).is_none());
            acc
        });

    for (other_label, train_valid_resources) in labels_others_train_valid_resources.iter_mut() {
        for (label, resources) in &labels_train_and_valid_sets {
            if label != other_label {
                train_valid_resources.0.extend(resources.0.iter());
                train_valid_resources.1.extend(resources.1.iter());
            }
        }
    }

    let labels_others_train_valid_resources = Arc::new(labels_others_train_valid_resources.into_iter().map(|(label, resources)| {
        let mut train_resources = resources.0;
        train_resources.shuffle(rng);
        train_resources.truncate(labels_train_and_valid_sets.get(&label).expect(&format!("no {label} for train resources")).0.len());

        let mut valid_resources = resources.1;
        valid_resources.shuffle(rng);
        valid_resources.truncate(labels_train_and_valid_sets.get(&label).expect(&format!("no {label} for valid resources")).1.len());

        (
            label,
            (
                train_resources.into_iter().cloned().collect::<Vec<_>>(),
                valid_resources.into_iter().cloned().collect::<Vec<_>>(),
            ),
        )
    }).collect::<HashMap<String, (Vec<Resource>, Vec<Resource>)>>());

    (labels_train_and_valid_sets, labels_others_train_valid_resources)
}

//TODO mean, std per channel
pub struct ResourceStats {
    size: Vec<u32>,
    length_dim: usize,
}

impl ResourceStats {
    pub fn new(size: Vec<u32>) -> Self {
        assert!(
            size.len() > 1,
            "should be two dimensions at least: channels x values, was {size:?}"
        );
        Self {
            size,
            length_dim: 1, //TODO param
        }
    }

    pub fn length(&self) -> u32 {
        self.size[self.length_dim]
    }
}

#[async_trait]
pub trait ResourcesConfig {
    type R;
    async fn to_resources<S: ResourceStore + ?Sized + Send + Sync>(
        &self,
        store: &Arc<Box<S>>,
    ) -> Result<Self::R, ThoError>;

    async fn check<S: ResourceStore + ?Sized + Send + Sync>(&self, _store: &Arc<Box<S>>) -> Result<(), ThoError> {
        Err(ThoError::new("resource config check is not implemented"))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use super::{build_resource_maps_with_rng, Resource};

    fn resource(id: u32, file_stem: &str) -> Resource {
        Resource::new(id, 1, file_stem, 100)
    }

    #[test]
    fn test_build_resource_maps() {
        let train = vec![
            ("label_a".to_string(), vec![resource(1, "a_tr_1"), resource(2, "a_tr_2")]),
            ("label_b".to_string(), vec![resource(3, "b_tr_1"), resource(4, "b_tr_2")]),
            ("label_c".to_string(), vec![resource(5, "c_tr_1"), resource(6, "c_tr_2")]),
        ];
        let valid = vec![
            ("label_a".to_string(), vec![resource(11, "a_vl_1")]),
            ("label_b".to_string(), vec![resource(12, "b_vl_1")]),
            ("label_c".to_string(), vec![resource(13, "c_vl_1")]),
        ];

        let mut rng = StdRng::seed_from_u64(42);
        let (labels_train_and_valid_sets, labels_others_train_valid_resources) =
            build_resource_maps_with_rng(train.clone(), valid.clone(), &mut rng);


        assert_eq!(labels_train_and_valid_sets.get("label_a").expect("missing label_a").0, train[0].1);
        assert_eq!(labels_train_and_valid_sets.get("label_a").expect("missing label_a").1, valid[0].1);
        assert_eq!(labels_train_and_valid_sets.get("label_b").expect("missing label_b").0, train[1].1);
        assert_eq!(labels_train_and_valid_sets.get("label_b").expect("missing label_b").1, valid[1].1);
        assert_eq!(labels_train_and_valid_sets.get("label_c").expect("missing label_c").0, train[2].1);
        assert_eq!(labels_train_and_valid_sets.get("label_c").expect("missing label_c").1, valid[2].1);

        for (label, (label_train, label_valid)) in &labels_train_and_valid_sets {


            let (others_train, others_valid) = labels_others_train_valid_resources.get(label).expect("missing others");

            // debug!("test_build_resource_maps: label={label} \nlabel_train={label_train:?} \nlabel_valid={label_valid:?} \nothers_train={others_train:?} \nothers_valid={others_valid:?}");

            let other_train_pool = labels_train_and_valid_sets
                .iter()
                .filter(|(other_label, _)| *other_label != label)
                .flat_map(|(_, (train_resources, _))| train_resources.iter().cloned())
                .collect::<HashSet<_>>();
            let other_valid_pool = labels_train_and_valid_sets
                .iter()
                .filter(|(other_label, _)| *other_label != label)
                .flat_map(|(_, (_, valid_resources))| valid_resources.iter().cloned())
                .collect::<HashSet<_>>();

            let label_train_set = label_train.iter().cloned().collect::<HashSet<_>>();
            let label_valid_set = label_valid.iter().cloned().collect::<HashSet<_>>();
            let others_train_set = others_train.iter().cloned().collect::<HashSet<_>>();
            let others_valid_set = others_valid.iter().cloned().collect::<HashSet<_>>();

            assert_eq!(others_train.len(), label_train.len());
            assert_eq!(others_valid.len(), label_valid.len());
            assert_eq!(others_train_set.len(), others_train.len());
            assert_eq!(others_valid_set.len(), others_valid.len());
            assert!(others_train.iter().all(|resource| other_train_pool.contains(resource)));
            assert!(others_valid.iter().all(|resource| other_valid_pool.contains(resource)));
            assert!(others_train.iter().all(|resource| !label_train_set.contains(resource)));
            assert!(others_valid.iter().all(|resource| !label_valid_set.contains(resource)));
        }
    }
}
