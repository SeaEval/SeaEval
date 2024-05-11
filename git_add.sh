


git add .

git ls-files --stage | while read mode object stage file; do
  size=$(git cat-file -s $object)
  if [ "$size" -gt 40000000 ]; then
    echo "Unstaging $file, size: $size"
    git reset HEAD "$file"
  fi
done