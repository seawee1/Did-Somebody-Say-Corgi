LATEST_VERSION=$(curl --silent "https://github.com/mozilla/geckodriver/releases" | grep -E "releases/tag/(.+)\">" | sed -E 's/.*"([^"]+)".*/\1/' | head -1 | grep -E -o "v[0-9]+\.[0-9]+\.[0-9]+")
echo "Installing geckodriver $LATEST_VERSION"
wget wget "https://github.com/mozilla/geckodriver/releases/download/$LATEST_VERSION/geckodriver-$LATEST_VERSION-linux64.tar.gz"
sudo sh -c "tar -x geckodriver -zf geckodriver-$LATEST_VERSION-linux64.tar.gz -O > /usr/bin/geckodriver"
sudo chmod +x /usr/bin/geckodriver
rm "geckodriver-$LATEST_VERSION-linux64.tar.gz"