'use strict';
module.exports = (sequelize, DataTypes) => {
  const User = sequelize.define('User', {
    id: {
      type: DataTypes.UUID,
      primaryKey: true,
      defaultValue: DataTypes.UUIDV4
    },
    user_name: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: true
    },
    email: {
      type: DataTypes.STRING,
      allowNull: true,
      unique: true
    },
    first_name: {
      type: DataTypes.STRING,
      allowNull: true,
      unique: false
    },
    last_name: {
      type: DataTypes.STRING,
      allowNull: true,
      unique: false
    },
    complete: {
      type: DataTypes.BOOLEAN,
      allowNull: false,
      defaultValue: false
    },
    profile_picture: {
      type: DataTypes.STRING,
      defaultValue: "https://images.unsplash.com/photo-1578320339911-5e7974b2720a?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2136&q=80",
      allowNull: false
    },
    active_socket: {
      type: DataTypes.STRING,
      allowNull: true,
      unique: false
    }
  }, {});
  User.associate = function(models) {
    // User.hasOne(models.user_passwords, { foreignKey: 'userId' });
    // associations can be defined here
  };
  return User;
};