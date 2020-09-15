'use strict';
module.exports = (sequelize, DataTypes) => {
  const superlearners = sequelize.define('superlearners', {
    id: {
      type: DataTypes.UUID,
      primaryKey: true,
      defaultValue: DataTypes.UUIDV1,
    },
    projectId: {
      type: DataTypes.UUID,
      allowNull: false,
      unique: false,
    },
    location1: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: false,
    },
    location2: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: false,
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: false,
    }
  }, {});
  superlearners.associate = function(models) {
    // associations can be defined here
  };
  return superlearners;
};