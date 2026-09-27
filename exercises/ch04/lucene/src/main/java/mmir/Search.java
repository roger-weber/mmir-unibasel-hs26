package mmir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * A small Lucene 10 command-line search tool for the Chapter 4 exercise.
 *
 * Two subcommands:
 *   index --data <tsv> --dir <indexdir>
 *   query --dir <indexdir> --q "<text>" --k <n> [--year-min <year>] [--fuzzy]
 *
 * The TSV data file has one document per line, tab-separated:
 *   id  <TAB>  title  <TAB>  text  <TAB>  year
 *
 * `index` prints {"indexed": n}; `query` prints a JSON array of
 * {"id", "title", "score", "year"} objects, best match first.
 *
 * You implement exactly two methods: buildDocument(...) and buildQuery(...).
 * Everything else (argument parsing, reading the data, running the search, and
 * printing JSON) is already written for you.
 */
public class Search {

    /** One analyzer is used for BOTH indexing and querying, as Lucene requires. */
    static final Analyzer ANALYZER = new StandardAnalyzer();

    // Field names, shared by the index side (buildDocument) and the query/display side
    // (buildQuery, doQuery), so the two never drift apart.
    static final String FIELD_ID = "id";
    static final String FIELD_TITLE = "title";
    static final String FIELD_TEXT = "text";
    static final String FIELD_YEAR = "year";

    // ─────────────────────────────────────────────────────────────────────────
    // YOUR TASK 1: turn one data record into a Lucene document.
    //
    // Return a Document with these fields:
    //   - "id":    the id, stored, matched exactly (not tokenized)
    //   - "title": the title, tokenized and stored
    //   - "text":  the searchable text, tokenized (storing it is not required)
    //   - "year":  the publication year, as a numeric field that supports range
    //              queries, and stored
    // Use the FIELD_* constants above for the field names, so the index side matches the
    // query and display sides. Choose the Lucene field type that fits each column.
    // ─────────────────────────────────────────────────────────────────────────
    static Document buildDocument(String id, String title, String text, int year) {
        // YOUR CODE HERE
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // YOUR TASK 2: build the query.
    //
    // Parse the free-text query `q` against the "text" field with the given
    // analyzer, so the query is analyzed the same way the documents were.
    //   - If `fuzzy` is true, each word of the query should match approximately,
    //     so a small spelling mistake still finds the document.
    //   - If `yearMin` is not null, the result must be restricted to documents
    //     whose "year" is at least yearMin. This restriction should not change the
    //     relevance scores of the documents that pass it.
    // Return the Query object; the caller runs it and prints the top hits.
    // ─────────────────────────────────────────────────────────────────────────
    static Query buildQuery(String q, Integer yearMin, boolean fuzzy, Analyzer analyzer)
            throws ParseException {
        // YOUR CODE HERE
        throw new UnsupportedOperationException("TODO: implement this");
    }

    // ─── Provided: build the index from the TSV file ──────────────────────────
    static void doIndex(Path dataFile, Path indexDir) throws Exception {
        try (Directory dir = FSDirectory.open(indexDir);
             IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig(ANALYZER))) {
            writer.deleteAll();  // rebuild from scratch every run
            int count = 0;
            for (String line : Files.readAllLines(dataFile)) {
                if (line.isBlank()) {
                    continue;
                }
                String[] f = line.split("\t", -1);
                if (f.length < 4) {
                    continue;
                }
                int year;
                try {
                    year = Integer.parseInt(f[3].trim());
                } catch (NumberFormatException e) {
                    year = 0;
                }
                writer.addDocument(buildDocument(f[0], f[1], f[2], year));
                count++;
            }
            writer.commit();
            System.out.println("{\"indexed\": " + count + "}");
        }
    }

    // ─── Provided: run one search and print the hits as JSON ──────────────────
    static void doQuery(Path indexDir, String q, int k, Integer yearMin, boolean fuzzy)
            throws Exception {
        if (q == null || q.isBlank()) {
            System.out.println("[]");
            return;
        }
        try (Directory dir = FSDirectory.open(indexDir);
             IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            StoredFields stored = searcher.storedFields();
            Query query = buildQuery(q, yearMin, fuzzy, ANALYZER);
            TopDocs top = searcher.search(query, Math.max(1, k));

            StringBuilder sb = new StringBuilder("[");
            int i = 0;
            for (ScoreDoc sd : top.scoreDocs) {
                if (i >= k) {
                    break;
                }
                Document d = stored.document(sd.doc);
                if (i > 0) {
                    sb.append(", ");
                }
                String year = d.get(FIELD_YEAR);
                sb.append("{\"id\": \"").append(esc(d.get(FIELD_ID))).append("\", ")
                  .append("\"title\": \"").append(esc(d.get(FIELD_TITLE))).append("\", ")
                  .append("\"score\": ").append(String.format(Locale.ROOT, "%.6f", sd.score)).append(", ")
                  .append("\"year\": ").append(year == null ? "null" : year)
                  .append("}");
                i++;
            }
            sb.append("]");
            System.out.println(sb);
        }
    }

    /** Escape a string for inclusion inside a JSON double-quoted value. */
    static String esc(String s) {
        if (s == null) {
            return "";
        }
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':  b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default:   b.append(c);
            }
        }
        return b.toString();
    }

    // ─── Provided: parse the command line and dispatch ────────────────────────
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("usage: (index --data <tsv> --dir <dir>) | "
                    + "(query --dir <dir> --q <text> --k <n> [--year-min <y>] [--fuzzy])");
            System.exit(2);
            return;
        }
        Map<String, String> opt = new HashMap<>();
        Set<String> flags = new HashSet<>();
        for (int i = 1; i < args.length; i++) {
            String a = args[i];
            if (a.startsWith("--")) {
                String key = a.substring(2);
                if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    opt.put(key, args[++i]);
                } else {
                    flags.add(key);
                }
            }
        }
        switch (args[0]) {
            case "index":
                doIndex(Path.of(opt.get("data")), Path.of(opt.get("dir")));
                break;
            case "query":
                Integer yearMin = opt.containsKey("year-min") ? Integer.valueOf(opt.get("year-min")) : null;
                int k = opt.containsKey("k") ? Integer.parseInt(opt.get("k")) : 10;
                doQuery(Path.of(opt.get("dir")), opt.getOrDefault("q", ""), k, yearMin, flags.contains("fuzzy"));
                break;
            default:
                System.err.println("unknown subcommand: " + args[0]);
                System.exit(2);
        }
    }
}
