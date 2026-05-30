use std::{
    fs, io,
    path::{Path, PathBuf},
};

#[derive(Debug, PartialEq, Eq)]
pub enum ServeConfig {
    Explicit(PathBuf),
    Candidates(Vec<PathBuf>),
}

pub fn prepare_serve_config(config: Option<PathBuf>, dir: &Path) -> io::Result<ServeConfig> {
    if let Some(config) = config {
        return Ok(ServeConfig::Explicit(config));
    }

    let candidates = discover_config_candidates(dir)?;
    if candidates.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no config.*.json found in the current directory; pass --config <FILE> to choose one explicitly",
        ));
    }

    Ok(ServeConfig::Candidates(candidates))
}

pub fn discover_config_candidates(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut candidates = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }

        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("config.")
            && name.ends_with(".json")
            && name.len() > "config.".len() + ".json".len()
        {
            candidates.push(entry.path());
        }
    }

    candidates.sort_by_key(|path| {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    Ok(candidates)
}

pub fn parse_config_selection(input: &str, count: usize) -> Result<usize, &'static str> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(0);
    }

    let selected = trimmed
        .parse::<usize>()
        .map_err(|_| "Please enter a number from the list.")?;
    if (1..=count).contains(&selected) {
        Ok(selected - 1)
    } else {
        Err("Selection out of range. Try one of the listed numbers.")
    }
}

pub fn stats_endpoint(url: &str) -> Result<String, io::Error> {
    let trimmed = url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "stats URL cannot be empty",
        ));
    }

    if trimmed.ends_with("/stats") {
        Ok(trimmed.to_owned())
    } else {
        Ok(format!("{trimmed}/stats"))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, io};

    use tempfile::tempdir;

    use super::{
        ServeConfig, discover_config_candidates, parse_config_selection, prepare_serve_config,
        stats_endpoint,
    };

    #[test]
    fn discover_config_candidates_only_returns_matching_json_files() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.z.json"), "{}").unwrap();
        fs::write(dir.path().join("config.a.json"), "{}").unwrap();
        fs::write(dir.path().join("config.json"), "{}").unwrap();
        fs::write(dir.path().join("config.a.toml"), "").unwrap();
        fs::create_dir(dir.path().join("config.dir.json")).unwrap();

        let files = discover_config_candidates(dir.path()).unwrap();
        let names: Vec<_> = files
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert_eq!(names, vec!["config.a.json", "config.z.json"]);
    }

    #[test]
    fn prepare_serve_config_returns_explicit_config_without_discovery() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("config.example.json");

        let resolved = prepare_serve_config(Some(config.clone()), dir.path()).unwrap();

        assert_eq!(resolved, ServeConfig::Explicit(config));
    }

    #[test]
    fn prepare_serve_config_returns_sorted_candidates() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.z.json"), "{}").unwrap();
        fs::write(dir.path().join("config.a.json"), "{}").unwrap();

        let resolved = prepare_serve_config(None, dir.path()).unwrap();

        assert_eq!(
            resolved,
            ServeConfig::Candidates(vec![
                dir.path().join("config.a.json"),
                dir.path().join("config.z.json"),
            ])
        );
    }

    #[test]
    fn prepare_serve_config_errors_when_no_candidates_exist() {
        let dir = tempdir().unwrap();

        let err = prepare_serve_config(None, dir.path()).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(err.to_string().contains("no config.*.json found"));
    }

    #[test]
    fn parse_config_selection_supports_default_and_valid_indexes() {
        assert_eq!(parse_config_selection("", 3).unwrap(), 0);
        assert_eq!(parse_config_selection("2", 3).unwrap(), 1);
    }

    #[test]
    fn parse_config_selection_rejects_bad_input() {
        assert_eq!(
            parse_config_selection("abc", 3).unwrap_err(),
            "Please enter a number from the list."
        );
        assert_eq!(
            parse_config_selection("4", 3).unwrap_err(),
            "Selection out of range. Try one of the listed numbers."
        );
    }

    #[test]
    fn stats_endpoint_appends_stats_path() {
        assert_eq!(
            stats_endpoint("http://127.0.0.1:8800").unwrap(),
            "http://127.0.0.1:8800/stats"
        );
        assert_eq!(
            stats_endpoint("http://127.0.0.1:8800/").unwrap(),
            "http://127.0.0.1:8800/stats"
        );
    }

    #[test]
    fn stats_endpoint_accepts_full_endpoint() {
        assert_eq!(
            stats_endpoint("http://127.0.0.1:8800/stats").unwrap(),
            "http://127.0.0.1:8800/stats"
        );
    }

    #[test]
    fn stats_endpoint_rejects_empty_url() {
        let err = stats_endpoint("   ").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn probe_config_update_writes_capability_fields() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.probe.json");
        fs::write(
            &path,
            r#"{
                "proxy_host": "127.0.0.1",
                "proxy_port": 8800,
                "proxy_api_key": "key",
                "upstream_base_url": "https://api.example.com",
                "upstream_model": "m",
                "upstream_api_key": "k"
            }"#,
        )
        .unwrap();

        // Simulate what cmd_probe --write does
        let raw = fs::read_to_string(&path).unwrap();
        let mut doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let obj = doc.as_object_mut().unwrap();
        obj.insert(
            "upstream_supports_named_tool_choice".to_owned(),
            serde_json::json!(true),
        );
        obj.insert(
            "upstream_supports_tool_choice_required".to_owned(),
            serde_json::json!(false),
        );
        obj.insert(
            "upstream_supports_reasoning_content".to_owned(),
            serde_json::json!(true),
        );
        obj.insert(
            "upstream_supports_reasoning_effort".to_owned(),
            serde_json::json!(true),
        );
        obj.insert(
            "upstream_supports_image_input".to_owned(),
            serde_json::json!(false),
        );
        let formatted = serde_json::to_string_pretty(&doc).unwrap();
        fs::write(&path, formatted + "\n").unwrap();

        // Verify the file is valid JSON and has the new fields
        let updated: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(updated["upstream_supports_named_tool_choice"], true);
        assert_eq!(updated["upstream_supports_tool_choice_required"], false);
        assert_eq!(updated["upstream_supports_reasoning_content"], true);
        assert_eq!(updated["upstream_supports_reasoning_effort"], true);
        assert_eq!(updated["upstream_supports_image_input"], false);
        // Original fields preserved
        assert_eq!(updated["proxy_host"], "127.0.0.1");
    }
}
